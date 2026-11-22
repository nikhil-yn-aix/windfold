import json
import math
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest
from scipy import integrate, stats

from windfold.evaluation import metrics
from windfold.schemas import PREDICTIONS
from windfold.smearing import SMEARING


def at(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2020, 1, day, hour, minute, tzinfo=UTC)


POWERS = {
    at(1): {"A": 1000.0, "B": 800.0, "C": 600.0},
    at(1, 0, 10): {"A": 900.0, "B": 950.0, "C": 700.0, "D": 1100.0},
    at(2, 13): {"B": 500.0, "C": 650.0},
}
WAKED = {"A": True, "B": False, "C": True, "D": False}


def prepared() -> pl.DataFrame:
    rows = []
    for t, powers in POWERS.items():
        logs = {name: math.log(p) for name, p in powers.items()}
        centre = sum(logs.values()) / len(logs)
        for name, p in powers.items():
            rows.append((t, name, p, logs[name] - centre, len(powers), WAKED[name]))
    frame = pl.DataFrame(
        rows,
        schema=["timestamp", "turbine", "power_kw", "y", "valid_turbines", "waked"],
        orient="row",
    )
    return frame.with_columns(
        valid_turbines=pl.col("valid_turbines").cast(pl.UInt8),
        speed_bin=pl.when(pl.col("turbine") == "A").then(1).otherwise(0).cast(pl.UInt8),
        direction_bucket=pl.lit(40, pl.UInt8),
        farm_direction=pl.lit(201.0),
        turbulence_intensity=pl.when(pl.col("turbine") == "B").then(None).otherwise(0.08),
        steady=pl.col("timestamp") == at(1),
    )


def predictions(shift: float = 0.05, variance: float | None = 0.01) -> pl.DataFrame:
    base = prepared()
    offsets = {"A": shift, "B": -shift / 2, "C": 0.02, "D": -0.01}
    return base.select(
        "timestamp",
        "turbine",
        "y",
        mean=pl.col("y") + pl.col("turbine").replace_strict(offsets),
        variance=pl.lit(variance, pl.Float64),
        support=pl.lit(None, pl.Int64),
        waked="waked",
    ).cast(PREDICTIONS)


def test_median_power_equals_peers_geometric_mean_times_predicted_contrast():
    rows = metrics.scored(predictions(), prepared())
    for row in rows.iter_rows(named=True):
        peers = [p for name, p in POWERS[row["timestamp"]].items() if name != row["turbine"]]
        geometric = math.exp(sum(math.log(p) for p in peers) / len(peers))
        m = row["valid_turbines"]
        contrast = m / (m - 1) * row["mean"]
        expected = geometric * math.exp(contrast)
        got = metrics.median_power(
            np.array([row["power_kw"]]), np.array([row["error"]]), np.array([row["scale"]])
        )[0]
        assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("error", "sd"), [(0.3, 0.2), (-1.1, 0.5), (0.0, 1.0), (2.0, 0.05)])
def test_gaussian_crps_matches_numerical_integration(error, sd):
    def integrand(x):
        return (stats.norm.cdf(x, 0.0, sd) - (x >= error)) ** 2

    numeric = integrate.quad(integrand, -40 * sd - abs(error), error, limit=200)[0]
    numeric += integrate.quad(integrand, error, 40 * sd + abs(error), limit=200)[0]
    assert metrics.gaussian_crps(np.array([error]), np.array([sd]))[0] == pytest.approx(
        numeric, rel=1e-6
    )
    assert metrics.gaussian_crps(np.array([error]), np.array([0.0]))[0] == abs(error)


def test_errors_sum_by_day_waked_and_speed_bin():
    rows = metrics.scored(predictions(), prepared())
    table = metrics.errors(rows, "farm", "model", "split", None)
    first_day_waked = table.filter((pl.col("day") == date(2020, 1, 1)) & pl.col("waked"))
    by_bin = {r["speed_bin"]: r for r in first_day_waked.iter_rows(named=True)}
    assert by_bin[1]["rows"] == 2 and by_bin[0]["rows"] == 2
    assert by_bin[1]["sse"] == pytest.approx(2 * 0.05**2)
    assert by_bin[0]["sse"] == pytest.approx(2 * 0.02**2)
    expected_kw = abs(1000 - 1000 * math.exp(1.5 * 0.05)) + abs(900 - 900 * math.exp(4 / 3 * 0.05))
    assert by_bin[1]["abs_error_kw"] == pytest.approx(expected_kw)
    crps_a = metrics.gaussian_crps(np.array([-1.5 * 0.05]), np.array([1.5 * 0.1]))[0]
    crps_a += metrics.gaussian_crps(np.array([-4 / 3 * 0.05]), np.array([4 / 3 * 0.1]))[0]
    assert by_bin[1]["crps"] == pytest.approx(crps_a)
    assert by_bin[1]["variance"] == pytest.approx(0.02)
    assert table["rows"].sum() == 9
    plain = metrics.errors(metrics.scored(predictions(variance=None), prepared()), "f", "m", "s", 2)
    assert (
        plain["crps"].null_count() == plain.height and plain["seed"].to_list() == [2] * plain.height
    )


def test_intervals_count_coverage_on_the_leave_one_out_scale():
    rows = metrics.scored(predictions(shift=0.31, variance=0.01), prepared())
    table = metrics.intervals(rows, [0.9, 0.99], "f", "m", "s", None)
    errors = (rows["scale"] * rows["error"]).abs().to_numpy()
    sd = (rows["scale"] * 0.1).to_numpy()
    for level in (0.9, 0.99):
        z = stats.norm.ppf((1 + level) / 2)
        part = table.filter(pl.col("level") == level)
        assert part["covered"].sum() == int((errors <= z * sd).sum())
        assert part["width"].sum() == pytest.approx(float((2 * z * sd).sum()))
        assert part["rows"].sum() == rows.height
    assert metrics.intervals(
        metrics.scored(predictions(variance=None), prepared()), [0.9], "f", "m", "s", None
    ).is_empty()


def test_breakdown_levels_by_hour_turbulence_and_arc():
    rows = metrics.scored(predictions(), prepared())
    table = metrics.breakdown(rows, [0.0, 0.06, 0.1, 1.0], 30.0, "f", "m", "s", None)
    hours = table.filter(pl.col("factor") == "hour").group_by("level").agg(pl.col("rows").sum())
    assert dict(hours.iter_rows()) == {0: 7, 13: 2}
    turbulence = table.filter(pl.col("factor") == "turbulence")
    assert set(turbulence["level"].to_list()) == {1} and turbulence["rows"].sum() == 6
    arcs = table.filter(pl.col("factor") == "arc")
    assert set(arcs["level"].to_list()) == {6} and arcs["rows"].sum() == 9
    steady = table.filter(pl.col("factor") == "steady").group_by("level").agg(pl.col("rows").sum())
    assert dict(steady.iter_rows()) == {1: 3, 0: 6}


def test_energy_ratio_sums_follow_the_flasc_definition():
    rows = metrics.scored(predictions(), prepared())
    record = json.loads(
        '{"smearing": [{"speed_bin": 0, "rows": 500, "factor": 1.01, "pooled": false},'
        ' {"speed_bin": 1, "rows": 120, "factor": 1.02, "pooled": true}]}'
    )
    smearing = pl.DataFrame(record["smearing"], schema=SMEARING)
    table = metrics.energy_ratio_sums(rows, prepared(), smearing, "f", "m", "s", None)
    observed, predicted, reference = 0.0, 0.0, 0.0
    for t, powers in POWERS.items():
        if "C" not in powers:
            continue
        free = [p for name, p in powers.items() if not WAKED[name] and name != "C"]
        if not free:
            continue
        row = rows.filter((pl.col("timestamp") == t) & (pl.col("turbine") == "C")).row(
            0, named=True
        )
        observed += powers["C"]
        predicted += powers["C"] * math.exp(-row["scale"] * row["error"]) * 1.01
        reference += sum(free) / len(free)
    turbine_c = table.filter(pl.col("turbine") == "C")
    assert turbine_c["power_kw"].sum() == pytest.approx(observed)
    assert turbine_c["predicted_kw"].sum() == pytest.approx(predicted)
    assert turbine_c["reference_kw"].sum() == pytest.approx(reference)
    turbine_b = table.filter(pl.col("turbine") == "B")
    assert turbine_b["reference_kw"].sum() == pytest.approx(1100.0)
    assert turbine_b["rows"].sum() == 1


def test_kendall_tau_b_matches_scipy_with_ties():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 6, size=(50, 9)).astype(float)
    y = rng.integers(0, 6, size=(50, 9)).astype(float)
    ours = metrics.kendall_tau_b(x, y)
    for k in range(50):
        assert ours[k] == pytest.approx(stats.kendalltau(x[k], y[k]).statistic)


def test_direction_weighted_mse_weights_arcs_by_their_timestamps():
    sse = np.array([[4.0, 9.0, 0.0]])
    rows = np.array([[2.0, 3.0, 0.0]])
    timestamps = np.array([[10.0, 30.0, 5.0]])
    expected = (10 * 2.0 + 30 * 3.0) / 40
    assert metrics.direction_weighted_mse(sse, rows, timestamps)[0] == pytest.approx(expected)


def test_a_prediction_without_a_prepared_row_fails():
    extra = predictions().vstack(
        pl.DataFrame(
            [(at(9), "A", 0.0, 0.0, None, None, True)], schema=dict(PREDICTIONS), orient="row"
        )
    )
    with pytest.raises(ValueError, match="no prepared row"):
        metrics.scored(extra, prepared())
