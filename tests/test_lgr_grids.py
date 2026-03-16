
from pathlib import Path

import numpy as np
import pytest
import xtgeo
from resdata.summary import Summary

from ccs_scripts.aggregate import grid3d_aggregate_map, grid3d_co2_mass_map
from ccs_scripts.aggregate._config import (
    AggregationMethod,
    CO2MassSettings,
    ComputeSettings,
    Input,
    Output,
    Property,
    RootConfig,
)


@pytest.fixture
def lgr_data_dir():
    return Path(__file__).parent / "lgr-model"


@pytest.fixture
def lgr_co2_mass_config(lgr_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
        ),
        output=Output(
            mapfolder=str(output_dir),
        ),
        co2_mass_settings=CO2MassSettings(
            unrst_source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
            init_source=str(lgr_data_dir / "DEP_GAS_4.INIT"),
            cirrus_info_file=str(lgr_data_dir / "DEP_GAS_4_INFO.csv"),
        ),
    )


@pytest.fixture
def lgr_aggregate_sgas_config(lgr_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
            properties=[
                Property(
                    source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
                    name="SGAS",
                )
            ],
        ),
        output=Output(
            mapfolder=str(output_dir),
        ),
        computesettings=ComputeSettings(
            aggregation=AggregationMethod.MEAN,
            zone=False,
        ),
    )


def test_mass_maps_with_lgr(lgr_data_dir, lgr_co2_mass_config):
    output_dir = Path(lgr_co2_mass_config.output.mapfolder)

    grid3d_co2_mass_map.generate_co2_mass_maps(lgr_co2_mass_config)
    # 9 time stamps, 3 maps per timestamp:
    assert len(list(Path(output_dir).glob("*.gri"))) == 9 * 3

    # Verify that the summed co2_mass_total maps match the UNSMRY FSMIP vector.
    # In this Cirrus model the equivalent of PFLOTRAN's FGMDS/FGMTR/FGMMO are
    # FSMDS (dissolved), FSMMO (mobile) and FSMTR (trapped).  Their sum must
    # equal FSMIP (total solvent mass in place).
    smry = Summary(str(lgr_data_dir / "DEP_GAS_4"))
    unsmry_totals = smry.numpy_vector("FSMIP", report_only=True)
    unsmry_totals_from_components = (
        smry.numpy_vector("FSMDS", report_only=True)
        + smry.numpy_vector("FSMMO", report_only=True)
        + smry.numpy_vector("FSMTR", report_only=True)
    )
    np.testing.assert_allclose(
        unsmry_totals_from_components,
        unsmry_totals,
        rtol=1e-6,
        err_msg="FSMDS + FSMMO + FSMTR should equal FSMIP",
    )
    date_to_unsmry = {
        dt.strftime("%Y%m%d"): total
        for dt, total in zip(smry.report_dates, unsmry_totals)
    }

    total_gri_files = sorted(Path(output_dir).glob("all--*co2_mass_total--*.gri"))
    assert len(total_gri_files) == 9
    for gri_path in total_gri_files:
        date_str = gri_path.stem.split("--")[-1]
        surface = xtgeo.surface_from_file(str(gri_path))
        gri_total = float(np.ma.filled(surface.values, 0.0).sum())
        unsmry_total = date_to_unsmry[date_str]
        np.testing.assert_allclose(
            gri_total,
            unsmry_total,
            rtol=0.01,
            err_msg=f"CO2 mass mismatch at {date_str}: gri={gri_total}, unsmry={unsmry_total}",
        )


def test_aggregate_maps_sgas_smooth_with_lgr(lgr_aggregate_sgas_config):
    """
    Verify that mean-SGAS aggregate maps produced from a grid containing LGR
    cells are smooth.  A non-smooth result (large jumps between adjacent map
    cells) indicates that the LGR section of the grid is being treated as
    inactive during aggregation, creating a hole where the refined cells are.
    """
    output_dir = Path(lgr_aggregate_sgas_config.output.mapfolder)

    grid3d_aggregate_map.generate_maps(
        lgr_aggregate_sgas_config.input,
        lgr_aggregate_sgas_config.zonation,
        lgr_aggregate_sgas_config.computesettings,
        lgr_aggregate_sgas_config.mapsettings,
        lgr_aggregate_sgas_config.output,
    )

    sgas_maps = sorted(output_dir.glob("all--mean_sgas--*.gri"))
    assert len(sgas_maps) == 9

    for gri_path in sgas_maps:
        surface = xtgeo.surface_from_file(str(gri_path))
        filled = np.ma.filled(surface.values, np.nan)
        max_adjacent_diff = max(
            np.nanmax(np.abs(np.diff(filled, axis=0))),
            np.nanmax(np.abs(np.diff(filled, axis=1))),
        )
        assert max_adjacent_diff < 0.1, (
            f"SGAS map {gri_path.name} is not smooth: "
            f"max adjacent cell difference = {max_adjacent_diff:.4f}. "
            "This likely means the LGR section is inactive during grid aggregation."
        )