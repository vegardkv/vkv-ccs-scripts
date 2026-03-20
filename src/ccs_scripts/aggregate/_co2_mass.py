import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import xtgeo
from resdata.resfile import ResdataFile
from resfo._unformatted.write import unformatted_write
from xtgeo.io._file import FileWrapper

from ccs_scripts.aggregate._config import CO2MassSettings
from ccs_scripts.co2_containment.co2_calculation import (
    Co2Data,
    Co2DataAtTimeStep,
    Scenario,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    fetch_properties,
    format_error,
    identify_gas_less_cells,
    identify_gas_less_cells_from_iterator,
    is_subset,
)

CO2_MASS_PNAME = "CO2Mass"

# pylint: disable=invalid-name,too-many-instance-attributes

# A single keyword entry as written to UNRST / EGRID files: (keyword_name, data).
_KwData = Union[List[np.int32], np.ndarray]
_Keyword = Tuple[str, _KwData]


@dataclass
class _MassData:
    unrst_path: List[str] = field(default_factory=list)
    unrst_kw: List[_Keyword] = field(default_factory=list)
    egrid_path: List[str] = field(default_factory=list)
    egrid_kw: List[_Keyword] = field(default_factory=list)


class MapName(Enum):
    MASS_TOT = "co2_mass_total"
    MASSDISW = "co2_mass_dissolved_water_phase"
    MASSDISO = "co2_mass_dissolved_oil_phase"
    MASS_GAS = "co2_mass_gas_phase"
    MASSTGAS = "co2_mass_trapped_gas_phase"
    MASSFGAS = "co2_mass_free_gas_phase"
    MigrationTime_MASS_TOT = "co2_mass_migration_time_total"


class PropertyGridOutput(TypedDict):
    data: np.ndarray
    unrst_path: str
    egrid_path: str


def _get_gasless(properties: xtgeo.GridProperties) -> np.ndarray:
    """
    Identifies global index for grid cells without CO2 based on Gas Saturation (SGAS)
    and Mole Fraction of Gas in dissolved phase (AMFG/XMF2)

    Args:
        properties (Dict) : Properties that will be used to compute CO2 mass

    Returns:
        np.ndarray
    """
    dissolved_prop = [d for d in ["AMFS", "AMFG", "XMF2"] if d in properties.names]
    if len(dissolved_prop) == 0 or "SGAS" not in properties.names:
        error_text = (
            "CO2 containment calculation failed. "
            "Cannot find required properties SGAS+AMFG, SGAS+XMF2 or SGAS+AMFS"
        )
        raise RuntimeError(format_error(error_text))

    return identify_gas_less_cells_from_iterator(
        (p.values for p in properties if p.name.startswith("SGAS")),
        (p.values for p in properties if p.name.startswith(dissolved_prop[0])),
    )


def _append_mass_step(
    mass_data: _MassData,
    seqnum: np.int32,
    intehead: np.ndarray,
    logihead: np.ndarray,
    kw_name: str,
    grid_output: PropertyGridOutput,
    custom_egrid: List[_Keyword],
) -> None:
    """Append one report-step's keywords to a mass_data accumulator."""
    mass_data.unrst_kw.extend(
        [
            ("SEQNUM  ", [seqnum]),
            ("INTEHEAD", intehead),
            ("LOGIHEAD", logihead),
            (kw_name, grid_output["data"]),
        ]
    )
    if grid_output["unrst_path"] not in mass_data.unrst_path:
        mass_data.unrst_path.append(grid_output["unrst_path"])
        mass_data.egrid_path.append(grid_output["egrid_path"])
        mass_data.egrid_kw.extend(custom_egrid)


def translate_co2data_to_property(
    co2_data: Co2Data,
    grid_file: str,
    co2_mass_settings: CO2MassSettings,
    grid_out_dir: str,
) -> List[Optional[str]]:
    """
    Convert CO2 data into 3D GridProperty

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in
                            each time step
        grid_file (str): Path to EGRID-file
        co2_mass_settings (CO2MassSettings): Settings from config file for calculation
                                             of CO2 mass maps.
        grid_out_dir (str): Path to store the produced 3D GridProperties.
        properties_to_extract (List): Names of the properties to be extracted

    Returns:
        List[List[xtgeo.GridProperty]]

    """
    timer = Timer()
    timer.start("translate_co2data_to_property")
    maps = co2_mass_settings.maps
    if maps is None:
        maps = []
    elif isinstance(maps, str):
        maps = [maps]
    maps = [map_name.lower() for map_name in maps]

    grid = xtgeo.grid_from_file(grid_file)
    store_all = "all" in maps or len(maps) == 0

    final_props: list[xtgeo.GridProperty] = []
    maps_to_generate = [MapName(m) for m in maps] if not store_all else [
        MapName.MASS_TOT,
        MapName.MASSDISW,
        MapName.MASSDISO,
        MapName.MASS_GAS,
        MapName.MASSTGAS,
        MapName.MASSFGAS,
    ]
    for co2_at_date in co2_data.data_list:
        tmp_props: dict[MapName, xtgeo.GridProperty] = _convert_to_grid(
            co2_at_date, grid, co2_data.active_cells, maps_to_generate
        )
        if store_all or "total_co2" in maps:
            final_props.append(tmp_props[MapName.MASS_TOT])
        if store_all or "dissolved_water_co2" in maps:
            final_props.append(tmp_props[MapName.MASSDISW])
        if (
            store_all or "dissolved_oil_co2" in maps
        ) and co2_data.scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            final_props.append(tmp_props[MapName.MASSDISO])
        if (
            store_all or "free_co2" in maps
        ) and not co2_mass_settings.residual_trapping:
            final_props.append(tmp_props[MapName.MASS_GAS])
        if (store_all or "free_co2" in maps) and co2_mass_settings.residual_trapping:
            final_props.append(tmp_props[MapName.MASSFGAS])
            final_props.append(tmp_props[MapName.MASSTGAS])

    # Write properties to files
    prop_paths: list[str] = []
    for p in final_props:
        file_path = Path(grid_out_dir) / f"{p.name}--{p.date}.grd"
        i = 0
        while file_path.exists() and i < 100000:
            file_path = Path(grid_out_dir) / f"{p.name}_{i}--{p.date}_.grd"
            i += 1
        p.to_file(file_path)
        prop_paths.append(str(file_path))

    timer.stop("translate_co2data_to_property")
    return prop_paths


def _create_custom_egrid_kw(
    grid_data: ResdataFile,
) -> List[_Keyword]:
    """
    Create the custom list of keywords to export the EGRID file for
    each co2_mass property
    """
    kw_sequence = [
        "FILEHEAD",
        "GRIDUNIT",
        "GDORIENT",
        "GRIDHEAD",
        "COORD   ",
        "ZCORN   ",
        "ACTNUM  ",
        "ENDGRID ",
        "NNCHEAD ",
        "NNC1    ",
        "NNC2    ",
    ]
    mandatory_kws = [
        "FILEHEAD",
        "GRIDUNIT",
        "GRIDHEAD",
        "COORD   ",
        "ZCORN   ",
        "ENDGRID ",
    ]
    custom_egrid = []
    for kw in kw_sequence:
        try:
            val = grid_data[kw.rstrip()][0].numpyView()
            custom_egrid.append((kw, val))
        except (AttributeError, ValueError, KeyError):
            try:
                val = grid_data[kw.rstrip()][0]
                custom_egrid.append((kw, val))
            except KeyError as err:
                if kw in mandatory_kws:
                    raise KeyError(
                        format_error(f"Mandatory key '{kw}' is missing in grid_data")
                    ) from err
                pass
    return custom_egrid


def _export_unrst_and_kw_data(mass_data: _MassData) -> Optional[str]:
    """
    Exports the grid with the property at different time steps as well as
    the path where the file is located

    Args:
        mass_data (_MassData): Accumulated mass data for one CO2 phase.

        Returns:
             Optional[str]
    """
    if len(mass_data.unrst_path) > 0:
        outfile_wrapper = FileWrapper(mass_data.unrst_path[0], mode="rb")
        with open(outfile_wrapper.file, "wb") as stream:
            unformatted_write(stream, mass_data.unrst_kw)
        grid_outfile_wrapper = FileWrapper(mass_data.egrid_path[0], mode="rb")
        with open(grid_outfile_wrapper.file, "wb") as stream:
            unformatted_write(stream, mass_data.egrid_kw)
        return mass_data.unrst_path[0]
    else:
        return None


def _get_gas_idxs(unrst_file: str) -> np.ndarray:
    """
    Gets the global index of cells with CO2

    Args:
        unrst_file (str): Path to UNRST-file
        properties_to_extract (List): Names of the properties to be extracted

    Returns:
        np.ndarray

    """
    props = xtgeo.gridproperties_from_file(unrst_file)
    gasless = _get_gasless(props)
    return gasless


def _convert_to_grid(
    co2_at_date: Co2DataAtTimeStep,
    grid: xtgeo.Grid,
    active_cells: np.ndarray,
) -> dict[MapName, xtgeo.GridProperty]:
    """
    Store CO2DataAtTimeStep for a property in a 3DGridProperties object

    Args:
        co2_at_date (Co2DataAtTimeStep):       Amount of CO2 per phase at each cell
                                               at each time step
        gas_idxs (np.ndarray):                 Global index of cells with CO2
        n_act_cells (int):                     Number of active cells in EGRID
        grid_out_dir (str):                    Path to store the produced
                                               3D GridProperties

    Returns:
        Dict[str, xtgeo.GridProperty]
    """

    def _create_prop(name: MapName, data: np.ndarray) -> xtgeo.GridProperty:
        prop = xtgeo.GridProperty(grid, name=name.value, date=co2_at_date.date)
        prop.values[active_cells] = data
        return prop

    props = {
        m: _create_prop(m, mass) for m, mass in [
            (MapName.MASS_TOT, co2_at_date.total_mass()),
            (MapName.MASSDISW, co2_at_date.dis_water_phase),
            (MapName.MASSDISO, co2_at_date.dis_oil_phase),
            (MapName.MASS_GAS, co2_at_date.gas_phase),
            (MapName.MASSTGAS, co2_at_date.trapped_gas_phase),
            (MapName.MASSFGAS, co2_at_date.free_gas_phase),
        ]
    }

    return props
