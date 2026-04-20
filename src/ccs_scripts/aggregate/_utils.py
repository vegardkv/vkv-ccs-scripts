import dataclasses
import getpass
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ccs_scripts.aggregate import _config
from ccs_scripts.aggregate._config import RootConfig
from ccs_scripts.utils._lgr_utils import (
    create_lgr_grid,
    extract_lgr_unrst,
    get_lgr_names,
)


def log_input_configuration(config_: RootConfig, map_type: str = "aggregate") -> None:
    """
    Log the provided input
    """
    version = "v0.16.0"
    is_dev_version = True
    if is_dev_version:
        version += "_dev"
        try:
            source_dir = os.path.dirname(os.path.abspath(__file__))
            short_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=source_dir
                )
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError:
            short_hash = "-"
        version += " (latest git commit: " + short_hash + ")"

    col1 = 37
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    if map_type == "aggregate":
        logging.info("CCS-scripts - Aggregate maps")
        logging.info("============================")
    elif map_type == "migration_time":
        logging.info("CCS-scripts - Migration time maps")
        logging.info("=================================")
    elif map_type == "co2_mass":
        logging.info("CCS-scripts - CO2 mass maps")
        logging.info("===========================")
    logging.info(f"{'Version':<{col1}} : {version}")
    logging.info(f"{'Date and time':<{col1}} : {date_time}")
    logging.info(f"{'User':<{col1}} : {getpass.getuser()}")
    logging.info(f"{'Host':<{col1}} : {socket.gethostname()}")
    logging.info(f"{'Platform':<{col1}} : {platform.system()} ({platform.release()})")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"{'Python version':<{col1}} : {py_version}")

    if map_type == "co2_mass":
        logging.info(f"\n{'Unit':<{col1}} : tons")

    logging.info("\nInput configuration:")
    logging.info(f"{'  Grid file':<{col1}} : {config_.input.grid}")
    if not os.path.isabs(config_.input.grid):
        logging.info(
            f"{'    => Absolute path':<{col1}} : "
            f"{os.path.abspath(config_.input.grid)}"
        )
    if map_type != "co2_mass":
        logging.info("  Properties:")
        if config_.input.properties is None:
            logging.info("    No properties specified")
        else:
            for p in config_.input.properties:
                logging.info(f"{'    - Name':<{col1}} : {p.name}")
                logging.info(
                    f"{'      Source':<{col1}} : "
                    f"{p.source if p.source is not None else '-'}"
                )
                logging.info(
                    f"{'      Lower threshold':<{col1}} : "
                    f"{p.lower_threshold if p.lower_threshold is not None else '-'}"
                )
    if len(config_.input.dates) > 0:
        logging.info(f"{'  Dates':<{col1}} : {', '.join(config_.input.dates)}")
    else:
        logging.info(f"{'  Dates':<{col1}} : - (not specified => using all dates)")

    op = config_.output
    logging.info("\nOutput configuration:")
    logging.info(f"{'  Map folder':<{col1}} : {op.mapfolder}")
    if not os.path.isabs(op.mapfolder):
        logging.info(
            f"{'    => Absolute path':<{col1}} : " f"{os.path.abspath(op.mapfolder)}"
        )
    if op.plotfolder is not None:
        logging.info(f"{'  Plot folder':<{col1}} : {op.plotfolder}")
        if not os.path.isabs(op.plotfolder):
            logging.info(
                f"{'    => Absolute path':<{col1}} : "
                f"{os.path.abspath(op.plotfolder)}"
            )
    else:
        logging.info(f"{'  Plot folder':<{col1}} : - (plot export not selected)")

    if map_type == "co2_mass":
        if op.gridfolder is not None:
            logging.info(f"{'  Grid folder':<{col1}} : {op.gridfolder}")
            if not os.path.isabs(op.gridfolder):
                logging.info(
                    f"{'    => Absolute path':<{col1}} : "
                    f"{os.path.abspath(op.gridfolder)}"
                )
        else:
            logging.info(
                f"{'  Grid folder':<{col1}} : - "
                f"(not specified, so temp exported 3D grid files will be deleted)"
            )
    else:
        logging.info(f"{'  Grid folder':<{col1}} : - (only relevant for co2 mass maps)")
    logging.info(
        f"{'  Use lower case in file names':<{col1}} : {_bool_str(op.lowercase)}"
    )
    logging.info(
        f"{'  Module/method for 2D plots':<{col1}} : "
        f"{'plotly library' if op.use_plotly else 'quickplot from xtgeoviz'}"
    )
    logging.info(
        f"{'  Add tag to file name for aggr. maps':<{col1}} : "
        f"{_bool_str(op.aggregation_tag)}"
    )
    logging.info(
        f"{'  Replace masked values with zeros':<{col1}} : "
        f"{_bool_str(op.replace_masked_with_zero)}"
    )

    logging.info("\nComputation configuration:")
    logging.info(
        f"{'  Aggregation method':<{col1}} : {config_.computesettings.aggregation.name}"
    )
    logging.info(
        f"{'  Weight by dz':<{col1}} : "
        f"{_bool_str(config_.computesettings.weight_by_dz)}"
    )
    logging.info(
        f"{'  Make maps for full grid (all zones)':<{col1}} : "
        f"{_bool_str(config_.computesettings.all)}"
    )
    logging.info(
        f"{'  Make maps per zone':<{col1}} : "
        f"{_bool_str(config_.computesettings.zone)}"
    )
    logging.info(
        f"{'  Calculate aggregate maps':<{col1}} : "
        f"{_bool_str(config_.computesettings.aggregate_map)}"
    )
    logging.info(
        f"{'  Calculate indicator maps':<{col1}} : "
        f"{_bool_str(config_.computesettings.indicator_map)}"
    )

    zon = config_.zonation
    logging.info("\nZonation configuration:")
    if not config_.computesettings.zone:
        logging.info(
            "(Note that these are not used since zone "
            "in computesettings is set to 'no')"
        )
    logging.info("  Z-property:")
    if zon.zproperty is None:
        logging.info("    No z-property specified")
    else:
        logging.info(f"{'    Source':<{col1}} : {zon.zproperty.source}")
        if not os.path.isabs(zon.zproperty.source):
            logging.info(
                f"{'      => Absolute path':<{col1}} : "
                f"{os.path.abspath(zon.zproperty.source)}"
            )
        logging.info(
            f"{'    Name':<{col1}} : "
            f"{zon.zproperty.name if zon.zproperty.name is not None else '-'}"
        )
        logging.info("    Zones:")
        zones = zon.zproperty.zones
        if len(zones) == 0:
            logging.info("      No zones specified")
        else:
            for z in zones:
                for i, (k, v) in enumerate(z.items()):
                    if i == 0:
                        logging.info(f"{f'      - {k}':<{col1}} : {v}")
                    else:
                        logging.info(f"{f'        {k}':<{col1}} : {v}")
    logging.info("  Z-ranges:")
    if len(zon.zranges) == 0:
        logging.info("    No z-ranges specified")
    else:
        for zr in zon.zranges:
            for i, (key, v2) in enumerate(zr.items()):
                if i == 0:
                    logging.info(f"{f'    - {key}':<{col1}} : {v2}")
                else:
                    logging.info(f"{f'      {key}':<{col1}} : {v2}")

    logging.info("\nMap configuration:")
    ms = config_.mapsettings
    if ms.templatefile is not None:
        logging.info("  Using template file (Option 1)")
    elif ms.xori is not None:
        logging.info("  Will use Option 2 since no template file has been specified")
    else:
        logging.info(
            "  Neither template file nor Origo x (etc) is specified,"
            " so will use pixel-to-cell-size ratio (Option 3)"
        )
    logging.info("  Option 1:")
    logging.info(
        f"{'    Template file':<{col1}} : "
        f"{ms.templatefile if ms.templatefile is not None else '- (not specified)'}"
    )
    if ms.templatefile is not None and not os.path.isabs(ms.templatefile):
        logging.info(
            f"{'      => Absolute path':<{col1}} : "
            f"{os.path.abspath(ms.templatefile)}"
        )
    logging.info("  Option 2:")
    logging.info(f"{'    Origo x':<{col1}} : {ms.xori if ms.xori is not None else '-'}")
    logging.info(f"{'    Origo y':<{col1}} : {ms.yori if ms.yori is not None else '-'}")
    logging.info(
        f"{'    Increment x':<{col1}} : {ms.xinc if ms.xinc is not None else '-'}"
    )
    logging.info(
        f"{'    Increment y':<{col1}} : {ms.yinc if ms.yinc is not None else '-'}"
    )
    logging.info(
        f"{'    Number of columns (x)':<{col1}} : "
        f"{ms.ncol if ms.ncol is not None else '-'}"
    )
    logging.info(
        f"{'    Number of rows (y)':<{col1}} : "
        f"{ms.nrow if ms.nrow is not None else '-'}"
    )
    if ms.xinc is not None and ms.ncol is not None:
        logging.info(f"{'    => Size x-direction':<{col1}} : {ms.xinc * ms.ncol}")
    if ms.yinc is not None and ms.nrow is not None:
        logging.info(f"{'    => Size y-direction':<{col1}} : {ms.yinc * ms.nrow}")
    logging.info("  Option 3:")
    logging.info(f"{'    Pixel-to-cell-size ratio':<{col1}} : {ms.pixel_to_cell_ratio}")

    cms = config_.co2_mass_settings
    if map_type == "co2_mass" and cms is not None:
        logging.info("\nCO2 mass configuration:")
        logging.info(f"{'  UNRST source':<{col1}} : {cms.unrst_source}")
        if not os.path.isabs(cms.unrst_source):
            logging.info(
                f"{'    => Absolute path':<{col1}} : "
                f"{os.path.abspath(cms.unrst_source)}"
            )
        logging.info(f"{'  INIT source':<{col1}} : {cms.init_source}")
        if not os.path.isabs(cms.init_source):
            logging.info(
                f"{'    => Absolute path':<{col1}} : "
                f"{os.path.abspath(cms.init_source)}"
            )
        txt = "(not specified => calculating all maps)"
        logging.info(
            f"{'  Maps to calculate':<{col1}} : "
            f"{cms.maps if cms.maps is not None else f'- {txt}'}"
        )
        logging.info(
            f"{'  Include residual trapping':<{col1}} : "
            f"{_bool_str(cms.residual_trapping)}"
        )
        logging.info(
            f"{'  Calculate migration time map':<{col1}} : "
            f"{_bool_str(cms.calculate_migration_time_map)}"
        )
        if cms.calculate_migration_time_map:
            threshold_text = (
                cms.migration_time_threshold
                if cms.migration_time_threshold is not None
                else "- (will be calculated automatically)"
            )
            logging.info(
                f"{'  Migration time threshold (tons)':<{col1}} : " f"{threshold_text}"
            )


def _bool_str(value: bool):
    return "yes" if value else "no"


# ---------------------------------------------------------------------------
# LGR preprocessing helpers
# ---------------------------------------------------------------------------


def _validate_lgr_name(lgr_name: str, grid_file: str | Path) -> None:
    """Raise ValueError if *lgr_name* is not found in *grid_file*."""
    available = get_lgr_names(Path(grid_file))
    if lgr_name not in available:
        names_str = ", ".join(available) if available else "none"
        raise ValueError(
            f"LGR '{lgr_name}' not found in grid '{grid_file}'. "
            f"Available LGRs: {names_str}"
        )


def prepare_for_lgr_processing_from_input(
    input_spec: _config.Input,
    lgr_name: str,
    tmp_dir: Path,
) -> _config.Input:
    if input_spec.properties is None:
        return input_spec  # No properties to prepare, so return original input spec

    unrst_props = [prop for prop in input_spec.properties if prop.source.endswith(".UNRST")]
    init_props = [prop for prop in input_spec.properties if prop.source.endswith(".INIT")]
    # TODO: warn about properties that are not from UNRST/INIT files
    unrst_file = unrst_props[0].source if unrst_props else None
    init_file = init_props[0].source if init_props else None
    lgr_egrid_file, lgr_unrst_file, lgr_init_file = prepare_for_lgr_processing(
        grid_file=input_spec.grid,
        properties_unrst_file=unrst_file,
        properties_init_file=init_file,
        lgr_name=lgr_name,
        tmp_dir=tmp_dir,
    )
    lgr_input = _config.Input(
        grid=str(lgr_egrid_file),
        properties=[
            dataclasses.replace(prop, source=str(lgr_unrst_file)) for prop in unrst_props
        ] + [
            dataclasses.replace(prop, source=str(lgr_init_file)) for prop in init_props
        ],
        dates=input_spec.dates,
    )
    return lgr_input


def prepare_for_lgr_processing(
    grid_file: Path,
    properties_unrst_file: Path | None,
    properties_init_file: Path | None,
    lgr_name: str,
    tmp_dir: Path,
) -> tuple[Path, Path | None, Path | None]:
    _validate_lgr_name(lgr_name, grid_file)
    lgr_egrid_file = tmp_dir / f"{lgr_name}.EGRID"
    create_lgr_grid(grid_file, lgr_name, lgr_egrid_file)

    if properties_unrst_file is not None:
        lgr_unrst_file = tmp_dir / f"{lgr_name}.UNRST"
        extract_lgr_unrst(properties_unrst_file, lgr_name, lgr_unrst_file)
    else:
        lgr_unrst_file = None

    if properties_init_file is not None:
        lgr_init_file = tmp_dir / f"{lgr_name}.INIT"
        extract_lgr_unrst(properties_init_file, lgr_name, lgr_init_file)
    else:
        lgr_init_file = None

    return lgr_egrid_file, lgr_unrst_file, lgr_init_file


def create_lgr_output(output: _config.Output, lgr_name: str) -> _config.Output:
    """Return a new Output with mapfolder set to output.mapfolder/lgr_name."""
    lgr_map_folder = Path(output.mapfolder) / lgr_name
    lgr_map_folder.mkdir(exist_ok=True, parents=True)
    return dataclasses.replace(output, mapfolder=str(lgr_map_folder))
