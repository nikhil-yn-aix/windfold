import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.config import load_farm
from windfold.readers.layout import east_north, read_layout
from windfold.schemas import LAYOUT

ROOT = Path(__file__).resolve().parents[2]


def test_east_north_matches_the_published_length_of_a_degree_at_50n():
    east, north = east_north(np.array([50.0, 50.01, 50.0]), np.array([0.0, 0.0, 0.01]))
    assert north[1] - north[0] == pytest.approx(1112.29, abs=0.5)
    assert east[2] - east[0] == pytest.approx(716.96, abs=0.5)
    convergence = np.radians(0.01 / 3) * np.sin(np.radians(50.0))
    assert east[1] - east[0] == pytest.approx(1112.29 * convergence, abs=0.002)
    assert (east.mean(), north.mean()) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_a_static_file_gives_turbines_in_order_centred_on_the_farm(tmp_path: Path):
    (tmp_path / "static.csv").write_bytes(
        "\ufeffTitle,Alternative Title,Hub Height (m),Latitude,Longitude,\r\n"
        "Farm 2,KWF2,68.5,52.402551,-0.949527,\r\n"
        "Farm 1,KWF1,78.5,52.400604,-0.947133,\r\n"
        ",,,,,\r\n".encode()
    )
    farm = load_farm("kelmarsh")
    farm = farm.model_copy(
        update={"source": farm.source.model_copy(update={"turbines": "static.csv"})}
    )
    layout = read_layout(farm, tmp_path)
    assert layout.schema == LAYOUT
    assert layout["turbine"].to_list() == ["KWF1", "KWF2"]
    assert layout["hub_height_m"].to_list() == [78.5, 68.5]
    assert layout.select(pl.col("x_m").sum(), pl.col("y_m").sum()).row(0) == pytest.approx(
        (0, 0), abs=1e-6
    )
    spacing = np.hypot(
        *(layout.select("x_m", "y_m").to_numpy()[1] - layout.select("x_m", "y_m").to_numpy()[0])
    )
    assert spacing == pytest.approx(271, abs=1.0)


def test_smarteole_turbines_come_from_the_package_and_hub_height_from_config(tmp_path: Path):
    folder = "SMARTEOLE-WFC-open-dataset/"
    with zipfile.ZipFile(tmp_path / "package.zip", "w") as package:
        package.writestr(
            folder + "SMARTEOLE_WakeSteering_Coordinates_staticData.csv",
            "Turbine,Latitude,Longitude,X_RGF93,Y_RGF93\r\n"
            "SMV2,49.84661,2.8015,685712,6971997\r\n"
            "SMV1,49.84975,2.80194,685745,6972347\r\n"
            "WindCube_v1,49.8450639,2.8013056,685698,6971825\r\n",
        )
    farm = load_farm("smarteole")
    farm = farm.model_copy(
        update={"source": farm.source.model_copy(update={"turbines": "package.zip"})}
    )
    layout = read_layout(farm, tmp_path)
    assert layout["turbine"].to_list() == ["SMV1", "SMV2"]
    assert layout["hub_height_m"].to_list() == [80.0, 80.0]


@pytest.mark.parametrize(
    ("farm", "pair", "metres"),
    [
        ("kelmarsh", ("KWF1", "KWF2"), 271),
        ("kelmarsh", ("KWF4", "KWF5"), 394),
    ],
)
def test_real_spacings_match_the_data_cards(farm: str, pair: tuple[str, str], metres: float):
    config = load_farm(farm)
    if not (ROOT / "data/raw" / farm / config.source.turbines).exists():
        pytest.skip(f"{farm} raw files are not on this machine")
    layout = read_layout(config, ROOT / "data/raw" / farm)
    a, b = (layout.filter(pl.col("turbine") == t).select("x_m", "y_m").to_numpy()[0] for t in pair)
    assert np.hypot(*(a - b)) == pytest.approx(metres, abs=1.0)
