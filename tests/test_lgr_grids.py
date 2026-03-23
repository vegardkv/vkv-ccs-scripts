
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
        computesettings=ComputeSettings(
            aggregation=AggregationMethod.DISTRIBUTE,
            zone=False,
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

    # In this Cirrus model, the following keywords are present:
    # - FSMDS (dissolved)
    # - FSMMO (mobile)
    # - FSMTR (trapped)
    # Their sum should equal FSMIP (total).
    # Create a dictionary of (date, property) -> value from the summary file for easy lookup:
    smry = Summary(str(lgr_data_dir / "DEP_GAS_4"))
    unsmry: dict[tuple[str, str], float] = {}
    for i, dt in enumerate(smry.report_dates):
        date_str = dt.strftime("%Y%m%d")
        for prop in ["FSMIP", "FSMDS", "FSMMO", "FSMTR"]:
            # Divide by 1000 for proper comparison
            unsmry[(date_str, prop)] = smry.numpy_vector(prop, report_only=True)[i] / 1000

    # Compare total amount of CO2 in total and dissolved maps to summary values. We allow
    # a 1% relative difference, which is somewhat arbitrary but should be sufficient to catch
    # major issues with the LGR handling.
    # TODO: look into comparing FSMMO as well
    total_gri_files = sorted(Path(output_dir).glob("all--*co2_mass_total--*.gri"))
    assert len(total_gri_files) == 9
    for gri_path in total_gri_files:
        date_str = gri_path.stem.split("--")[-1]
        surface = xtgeo.surface_from_file(str(gri_path))
        gri_total = float(np.ma.filled(surface.values, 0.0).sum())
        unsmry_total = unsmry[(date_str, "FSMIP")]
        assert gri_total == pytest.approx(unsmry_total, rel=0.01)

    dissolved_gri_files = sorted(
        Path(output_dir).glob("all--*co2_mass_dissolved_water_phase--*.gri")
    )
    assert len(dissolved_gri_files) == 9
    for gri_path in dissolved_gri_files:
        date_str = gri_path.stem.split("--")[-1]
        surface = xtgeo.surface_from_file(str(gri_path))
        gri_total = float(np.ma.filled(surface.values, 0.0).sum())
        fsmds_total = unsmry[(date_str, "FSMDS")]
        assert gri_total == pytest.approx(fsmds_total, rel=0.01)


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