import os
import shutil
import tempfile
from pathlib import Path

from resdata.resfile import FortIO, ResdataFile, openFortIO

from ccs_scripts.aggregate import grid3d_co2_mass_map


def adapt_reek_grid_for_co2_mass_map_test():
    """
    Adds the necessary properties to reek grid to make it usable for
    test_co2_mass_map_reek_grid
    """
    reek_unrstfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.UNRST"
    )
    properties = ResdataFile(str(reek_unrstfile))
    SGAS = properties["SGAS"]
    AMFG = []
    YMFG = []
    DGAS = []
    DWAT = []
    for x in SGAS:
        AMFG.append(x.copy())
        YMFG.append(x.copy())
        DGAS.append(x.copy())
        DWAT.append(x.copy())
    new_unrst_file = str(
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0-mass-maps.UNRST"
    )
    shutil.copy(str(reek_unrstfile), new_unrst_file)
    with openFortIO(new_unrst_file, mode=FortIO.APPEND_MODE) as f:
        for y in AMFG:
            y.name = "AMFG"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = a[i] * 0.02
            y.fwrite(f)
        for y in YMFG:
            y.name = "YMFG"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 0.99
            y.fwrite(f)
        for y in DGAS:
            y.name = "DGAS"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 100
            y.fwrite(f)
        for y in DWAT:
            y.name = "DWAT"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 1000
            y.fwrite(f)


def test_co2_mass_map_reek_grid():
    """
    Test CO2 mass maps generation, with eclipse Reek data
    """
    adapt_reek_grid_for_co2_mass_map_test()
    result = str(Path(__file__).absolute().parent / "answers" / "mass_map")
    if not os.path.exists(result):
        os.makedirs(result)
    grid3d_co2_mass_map.main(
        [
            "--config_co2_mass_map",
            str(
                Path(__file__).absolute().parent
                / "yaml"
                / "config_co2_mass_map_reek.yml"
            ),
            "--mapfolder",
            str(result),
        ]
    )
    dissolved_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_dissolved_water_phase--20010801.gri"
    )
    free_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_gas_phase--20010801.gri"
    )
    total_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_total--20010801.gri"
    )
    assert free_co2_file.exists()
    assert dissolved_co2_file.exists()
    assert total_co2_file.exists()
    shutil.rmtree(str(Path(__file__).absolute().parent / "answers" / "mass_map"))
    os.remove(
        str(
            Path(__file__).absolute().parent
            / "data"
            / "reek"
            / "eclipse"
            / "model"
            / "2_R001_REEK-0-mass-maps.UNRST"
        )
    )


def test_co2_mass_map_residual_trapping_cirrus():
    """
    Test CO2 mass maps, with synthetic_case cirrus data
    """
    result = str(Path(__file__).absolute().parent / "answers" / "mass_map")
    if not os.path.exists(result):
        os.makedirs(result)

    grid3d_co2_mass_map.main(
        [
            "--config_co2_mass_map",
            str(
                Path(__file__).absolute().parent
                / "yaml"
                / "config_co2_mass_map_cirrus.yml"
            ),
            "--mapfolder",
            str(result),
        ]
    )
    free_gas_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_free_gas_phase--23000101.gri"
    )
    trapped_gas_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_trapped_gas_phase--23000101.gri"
    )
    total_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_maps"
        / "all--co2_mass_total--23000101.gri"
    )
    assert free_gas_co2_file.exists()
    assert trapped_gas_co2_file.exists()
    assert not total_co2_file.exists()
    shutil.rmtree(str(Path(__file__).absolute().parent / "answers" / "mass_map"))


def test_mass_maps_with_lgr():
    from pathlib import Path

    from ccs_scripts.aggregate._config import CO2MassSettings, Input, Output, RootConfig

    data_dir = Path(__file__).parent / "lgr-model"

    with tempfile.TemporaryDirectory() as output_dir:
        config = RootConfig(
            input=Input(
                grid=str(data_dir / "DEP_GAS_4.EGRID"),
            ),
            output=Output(
                mapfolder=str(output_dir),
            ),
            co2_mass_settings=CO2MassSettings(
                unrst_source=str(data_dir / "DEP_GAS_4.UNRST"),
                init_source=str(data_dir / "DEP_GAS_4.INIT"),
                cirrus_info_file=str(data_dir / "DEP_GAS_4_INFO.csv"),
            ),
        )

        grid3d_co2_mass_map.generate_co2_mass_maps(config)
        # 9 time stamps, 3 maps per timestamp:
        assert len(list(Path(output_dir).glob("*.gri"))) == 9 * 3
