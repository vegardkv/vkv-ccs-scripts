from collections.abc import Iterable
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from resdata.grid import Grid
from resdata.resfile import ResdataFile

THRESHOLD_GAS = 1e-16
THRESHOLD_DISSOLVED = 1e-16  # Used also in co2_calculation to avoid numerical issues


def format_warning(txt: Union[str, Exception]) -> str:
    return f"\x1b[37;45m\x1b[1m{txt}\x1b[0m"


def format_error(txt: Union[str, Exception]) -> str:
    return f"\x1b[37;41m\x1b[1m{txt}\x1b[0m"


def log_saturation_summaries(props: Dict) -> None:
    sgas = props["SGAS"]
    swat = props["SWAT"]
    soil = props["SOIL"] if "SOIL" in props else None

    first_timestep = next(iter(sgas))
    saturations_first_timestep = [
        ("sgas", sgas[first_timestep]),
        ("swat", swat[first_timestep]),
    ]
    last_timestep = next(reversed(sgas))
    saturations_last_timestep = [
        ("sgas", sgas[last_timestep]),
        ("swat", swat[last_timestep]),
    ]

    if soil is not None:
        saturations_first_timestep.append(("soil", soil[first_timestep]))
        saturations_last_timestep.append(("soil", soil[last_timestep]))

    header = (
        f"\n{'Property':<15} {'Min':>12} {'P10':>12} "
        f"{'Median':>12} {'Mean':>12} {'P90':>12} {'Max':>12}"
    )
    logging.info("\nPhase saturation summaries for first timestep - Active cells only")
    logging.info(header)
    logging.info(f"{'-' * 93}")
    for label, values in saturations_first_timestep:
        row = (
            f"{label:<15} "
            f"{values.min():>12.1f} "
            f"{np.percentile(values, 10):>12.1f} "
            f"{np.median(values):>12.1f} "
            f"{values.mean():>12.1f} "
            f"{np.percentile(values, 90):>12.1f} "
            f"{values.max():>12.1f}"
        )
        logging.info(row)

    logging.info("\nPhase saturation summaries for last timestep - Active cells only")
    logging.info(header)
    logging.info(f"{'-' * 93}")

    for label, values in saturations_last_timestep:
        row = (
            f"{label:<15} "
            f"{values.min():>12.1f} "
            f"{np.percentile(values, 10):>12.1f} "
            f"{np.median(values):>12.1f} "
            f"{values.mean():>12.1f} "
            f"{np.percentile(values, 90):>12.1f} "
            f"{values.max():>12.1f}"
        )
        logging.info(row)


def test_for_soil(props: dict):
    if "SGAS" not in props or "SWAT" not in props:
        return None
    tol = 1e-6
    sgas = props["SGAS"]
    swat = props["SWAT"]
    soil = {}
    max_val = float("-inf")
    for date in sgas:
        soil[date] = np.maximum(0.0, 1.0 - sgas[date] - swat[date])
        max_soil_date = soil[date].max()
        if max_soil_date > max_val:
            max_val = max_soil_date
    return soil if max_val > tol else None


def fetch_properties(
    unrst: ResdataFile, props_to_extract: List
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """
    Fetches the properties in props_to_extract from a ResdataFile
    named unrst

    Args:
      unrst (ResdataFile): ResdataFile to fetch props_to_extract from
      props_to_extract: List with property names to be fetched

    Returns:
      Tuple

    """
    report_dates = unrst.report_dates
    props: dict[str, dict[str, np.ndarray]] = {}
    dates_with_missing_data: list[str] = []
    for p in props_to_extract:
        if not unrst.has_kw(p):
            # Ignore parameters not found in unrst. Parameters will
            # typically include static properties which are found in
            # the INIT file, but these are not relevant in this
            # context.
            continue
        props[p] = {}
        for d in unrst.report_dates:
            d_formatted = d.strftime("%Y%m%d")
            try:
                # We fetch via restart_get_kw, since this also works
                # for LGR models.
                kw = unrst.restart_get_kw(p, d)
            except IndexError:
                # If the property is not defined for this date in the
                # UNRST file, log the error and continue
                dates_with_missing_data.append(d_formatted)
                continue

            props[p][d_formatted] = kw.numpy_copy()
    if dates_with_missing_data:
        # Raise exception in case of error. An alternative is to
        # remove dates with missing data, but that is less
        # transparent.
        raise ValueError(
            format_error(
                f"At least one of the properties is missing data for "
                f"the following dates: {', '.join(dates_with_missing_data)}"
            )
        )

    if "SOIL" not in props:
        soil = test_for_soil(props)
        if soil is not None:
            props["SOIL"] = soil
            logging.info(
                "Oil Saturation (SOIL) not found as property"
                "\nHowever, as SGAS + SWAT is not 1 everywhere"
                "\nThe remaining saturation is assumed to be SOIL."
                "\nThis propery has been computed"
            )
        else:
            logging.info(
                "Oil Saturation is zero everywhere."
                "\n Therefore, two-phase scenario is assumed."
            )
    logging.info(
        "Done reading properties from file"
        "\nRelevant properties extracted:"
        f"\n    {', '.join(list(props.keys()))}\n"
    )
    return props, [d.strftime("%Y%m%d") for d in report_dates]


def identify_gas_less_cells(
    sgas: dict, dissolved_prop: Optional[dict] = None
) -> np.ndarray:
    """
    Identifies those cells that do not have gas. This is done based on thresholds for
    SGAS and AMFG/XMF2 (dissolved property).

    Args:
      sgas (dict): The values of SGAS for each grid cell
      dissolved_prop (dict): The values of AMFG or XMF2 for each grid cell

    Returns:
      np.ndarray

    """
    return identify_gas_less_cells_from_iterator(
        sgas.values(),
        dissolved_prop.values() if dissolved_prop is not None else None,
    )


def identify_gas_less_cells_from_iterator(
    sgas_iter: Iterable[np.ndarray],
    dissolved_iter: Iterable[np.ndarray] | None
) -> np.ndarray:
    gas_less = np.logical_and.reduce([np.abs(s) < THRESHOLD_GAS for s in sgas_iter])
    if dissolved_iter is not None:
        gas_less &= np.logical_and.reduce(
            [np.abs(d) < THRESHOLD_DISSOLVED for d in dissolved_iter]
        )
    return gas_less


def reduce_properties(
    properties: Dict[str, Dict[str, np.ndarray]], keep_idx: np.ndarray
) -> Dict:
    """
    Reduces the data of given properties by indices in keep_idx

    Args:
      properties (Dict): Data with values of properties
      keep_idx (np.ndarray): Which indices are retained

    Returns:
      Dict

    """
    return {
        p: {d: properties[p][d][keep_idx] for d in properties[p]} for p in properties
    }


def is_subset(first: List[str], second: List[str]) -> bool:
    """
    Determines if the elements of a list (first) are part of
    another list (second)

    Args:
      first (List): The list whose elements are searched in second
      second (List): The list where elements of first are searched

    Returns:
      bool

    """
    return all(x in second for x in first)


def find_active_and_gasless_cells(
    grid: Grid, properties, do_logging: bool = False, ignore_dissolved: bool = False
):
    act_num = grid.export_actnum().numpy_copy()
    active = np.where(act_num > 0)[0]

    if ignore_dissolved:
        gasless = identify_gas_less_cells(properties["SGAS"])
    else:
        dissolved_prop = None
        if is_subset(["SGAS", "AMFS"], list(properties.keys())):
            dissolved_prop = "AMFS"
        elif is_subset(["SGAS", "AMFG"], list(properties.keys())):
            dissolved_prop = "AMFG"
        elif is_subset(["SGAS", "XMF2"], list(properties.keys())):
            dissolved_prop = "XMF2"

        if dissolved_prop is not None:
            gasless = identify_gas_less_cells(
                properties["SGAS"], properties[dissolved_prop]
            )
        else:
            error_text = (
                "CO2 containment calculation failed. Cannot find required properties "
            )
            error_text += "SGAS+AMFG, SGAS+XMF2 or SGAS+AMFS"
            raise RuntimeError(format_error(error_text))

    if do_logging:
        logging.info(f"Number of grid cells                    : {len(act_num):>10}")
        logging.info(f"Number of active grid cells             : {len(active):>10}")
        logging.info(
            f"Number of active non-gasless grid cells : {len(active[~gasless]):>10}"
        )

    return active, gasless


def read_yaml_file(
    file_name: str,
) -> Dict:
    with open(file_name, "r", encoding="utf8") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            logging.error(format_error(exc))
            sys.exit(1)
