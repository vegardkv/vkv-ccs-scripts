from pathlib import Path

import numpy as np
import pytest
import xtgeo
from resdata.summary import Summary

from ccs_scripts.aggregate import (
    grid3d_aggregate_map,
    grid3d_co2_mass_map,
    grid3d_migration_time,
)
from ccs_scripts.aggregate._config import (
    AggregationMethod,
    CO2MassSettings,
    ComputeSettings,
    Input,
    LgrMapSettings,
    MapSettings,
    MigrationTimeSettings,
    Output,
    Property,
    RootConfig,
)

LGR_NAME = "rfin"


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


@pytest.fixture
def lgr_aggregate_config(lgr_data_dir, tmp_path):
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
        lgr_settings=[LgrMapSettings(name=LGR_NAME)],
    )


@pytest.fixture
def lgr_co2_mass_lgr_config(lgr_data_dir, tmp_path):
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
        lgr_settings=[LgrMapSettings(name=LGR_NAME)],
    )


@pytest.fixture
def lgr_migration_time_config(lgr_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
            properties=[
                Property(
                    source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
                    name="SGAS",
                    lower_threshold=0.001,
                )
            ],
        ),
        output=Output(
            mapfolder=str(output_dir),
        ),
        computesettings=ComputeSettings(
            zone=False,
        ),
        migration_time_settings=MigrationTimeSettings(),
        lgr_settings=[LgrMapSettings(name=LGR_NAME)],
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
    # dict[(date, property)] -> value from the summary file for easy lookup:
    smry = Summary(str(lgr_data_dir / "DEP_GAS_4"))
    unsmry: dict[tuple[str, str], float] = {}
    for i, dt in enumerate(smry.report_dates):
        date_str = dt.strftime("%Y%m%d")
        for prop in ["FSMIP", "FSMDS", "FSMMO", "FSMTR"]:
            # Divide by 1000 for proper comparison
            unsmry[(date_str, prop)] = (
                smry.numpy_vector(prop, report_only=True)[i] / 1000
            )

    # Compare total amount of CO2 in total and dissolved maps to summary
    # values. We allow a 1% relative difference, which is somewhat arbitrary
    # but should be sufficient to catch major issues with the LGR handling.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_maps_nonempty(gri_files):
    """
    For each .gri surface: assert unmasked values are all finite, and that the
    combined absolute sum across all files is positive (i.e. not all-zero).
    """
    total_abs_sum = 0.0
    for gri_path in gri_files:
        surface = xtgeo.surface_from_file(str(gri_path))
        unmasked = surface.values.compressed()
        assert np.all(np.isfinite(unmasked)), (
            f"Map {gri_path.name} contains NaN or Inf values"
        )
        total_abs_sum += float(np.sum(np.abs(unmasked)))
    assert total_abs_sum > 0.0, (
        "All LGR maps are zero across every time step — "
        "check that the LGR contains active CO2/property data"
    )


# ---------------------------------------------------------------------------
# Per-LGR map generation tests (lgr_settings code path)
# ---------------------------------------------------------------------------


def test_aggregate_maps_with_lgr_settings(lgr_aggregate_config):
    """
    generate_from_config with lgr_settings produces per-LGR aggregate maps
    under <output>/<lgr_name>/ alongside the parent-grid maps in <output>/.
    Maps are checked for absence of NaN/Inf and must not be uniformly zero.
    """
    output_dir = Path(lgr_aggregate_config.output.mapfolder)
    lgr_output_dir = output_dir / LGR_NAME

    grid3d_aggregate_map.generate_from_config(lgr_aggregate_config)

    parent_maps = sorted(output_dir.glob("*.gri"))
    lgr_maps = sorted(lgr_output_dir.glob("*.gri"))

    assert lgr_output_dir.is_dir()
    assert len(parent_maps) == 9, f"Expected 9 parent maps, got {len(parent_maps)}"
    assert len(lgr_maps) == 9, f"Expected 9 LGR maps, got {len(lgr_maps)}"
    _assert_maps_nonempty(lgr_maps)


def test_co2_mass_maps_with_lgr_settings(lgr_co2_mass_lgr_config):
    """
    generate_co2_mass_maps with lgr_settings produces per-LGR CO2 mass maps
    under <output>/<lgr_name>/ alongside the parent-grid maps in <output>/.
    Maps are checked for absence of NaN/Inf and must not be uniformly zero.
    """
    output_dir = Path(lgr_co2_mass_lgr_config.output.mapfolder)
    lgr_output_dir = output_dir / LGR_NAME

    grid3d_co2_mass_map.generate_co2_mass_maps(lgr_co2_mass_lgr_config)

    parent_maps = sorted(output_dir.glob("*.gri"))
    lgr_maps = sorted(lgr_output_dir.glob("*.gri"))

    assert lgr_output_dir.is_dir()
    # 9 timestamps × 3 mass types (total, free, dissolved)
    assert len(parent_maps) == 9 * 3, f"Expected 27 parent maps, got {len(parent_maps)}"
    assert len(lgr_maps) == 9 * 3, f"Expected 27 LGR maps, got {len(lgr_maps)}"
    _assert_maps_nonempty(lgr_maps)


def test_migration_time_maps_with_lgr_settings(lgr_migration_time_config):
    """
    generate_from_config (migration time) with lgr_settings produces a per-LGR
    migration time map under <output>/<lgr_name>/ alongside the parent-grid map
    in <output>/. Maps are checked for absence of NaN/Inf and must not be
    uniformly zero.
    """
    output_dir = Path(lgr_migration_time_config.output.mapfolder)
    lgr_output_dir = output_dir / LGR_NAME

    grid3d_migration_time.generate_from_config(lgr_migration_time_config)

    parent_maps = sorted(output_dir.glob("*.gri"))
    lgr_maps = sorted(lgr_output_dir.glob("*.gri"))

    assert lgr_output_dir.is_dir()
    # One migration time map per property (no per-date output)
    assert len(parent_maps) == 1, f"Expected 1 parent map, got {len(parent_maps)}"
    assert len(lgr_maps) == 1, f"Expected 1 LGR map, got {len(lgr_maps)}"
    _assert_maps_nonempty(lgr_maps)


# ---------------------------------------------------------------------------
# Invalid-input tests
# ---------------------------------------------------------------------------


def test_lgr_invalid_inputs(lgr_data_dir, tmp_path):
    """
    1. An unrecognised LGR name raises ValueError before any file I/O.
    2. A map template that does not overlap the LGR grid produces output files,
       but every cell value is zero (the silent-failure mode documented in
       _grid_aggregation.py when no grid cells fall inside the template).
    """
    # --- 1. Invalid LGR name -------------------------------------------
    output_bad_name = tmp_path / "output_bad_name"
    output_bad_name.mkdir()
    config_bad_name = RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
            properties=[
                Property(
                    source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
                    name="SGAS",
                )
            ],
        ),
        output=Output(mapfolder=str(output_bad_name)),
        computesettings=ComputeSettings(zone=False),
        lgr_settings=[LgrMapSettings(name="NONEXISTENT_LGR")],
    )
    with pytest.raises(ValueError, match="NONEXISTENT_LGR"):
        grid3d_aggregate_map.generate_from_config(config_bad_name)

    # --- 2. Non-overlapping map template ---------------------------------
    # The rfin LGR bounding box is roughly x: 1400–1800, y: 1800–2200.
    # Placing the template at x: 0–50, y: 0–50 guarantees zero overlap.
    output_nover = tmp_path / "output_nover"
    output_nover.mkdir()
    config_nover = RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
            properties=[
                Property(
                    source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
                    name="SGAS",
                )
            ],
        ),
        output=Output(mapfolder=str(output_nover)),
        computesettings=ComputeSettings(zone=False),
        lgr_settings=[
            LgrMapSettings(
                name=LGR_NAME,
                mapsettings=MapSettings(
                    xori=0.0, yori=0.0, xinc=10.0, yinc=10.0, ncol=5, nrow=5
                ),
            )
        ],
    )
    grid3d_aggregate_map.generate_from_config(config_nover)

    lgr_nover_maps = sorted((output_nover / LGR_NAME).glob("*.gri"))
    assert len(lgr_nover_maps) > 0, "Expected map files even for non-overlapping template"
    for gri_path in lgr_nover_maps:
        surface = xtgeo.surface_from_file(str(gri_path))
        assert np.all(np.ma.filled(surface.values, 0.0) == 0.0), (
            f"Map {gri_path.name} should be all-zero for a non-overlapping template"
        )

