import json
import shutil
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold import inflow
from windfold.analyses.inflow_smoothing import inflow_smoothing, smoothed

START = datetime(2020, 3, 1, tzinfo=UTC)
READINGS = [0.10, 0.20, 0.15, 0.12, 0.30, None, 0.08, 0.0, 0.11, float("nan"), 0.25, 0.09]


def rows_of(readings: list[float | None], turbines: int = 2) -> pl.DataFrame:
    """timestamps every 30 minutes, the first four in the calibration period, each with
    turbines rows sharing the timestamp's reading."""
    times = [START + timedelta(minutes=30 * k) for k in range(len(readings))]
    periods = ["calibration"] * 4 + ["development"] * (len(readings) - 4)
    return pl.DataFrame(
        {
            "timestamp": [t for t in times for _ in range(turbines)],
            "turbine": [f"T{i}" for _ in times for i in range(turbines)],
            "period": [p for p in periods for _ in range(turbines)],
            "turbulence_intensity": pl.Series(
                [r for r in readings for _ in range(turbines)], dtype=pl.Float64
            ),
        }
    )


def brute_force(readings: list[float | None], before: float, after: float) -> list[float]:
    """each measured reading's geometric mean over the measured readings of its period within
    [t - before, t + after] hours; others as they were, a missing one as nan."""
    times = [30 * k / 60 for k in range(len(readings))]
    periods = [k < 4 for k in range(len(readings))]
    value = [np.nan if r is None else r for r in readings]
    measured = [np.isfinite(v) and v > 0 for v in value]
    out = []
    for k, (t, own) in enumerate(zip(times, value, strict=True)):
        if not measured[k]:
            out.append(own)
            continue
        window = [
            np.log(value[j])
            for j in range(len(times))
            if measured[j] and periods[j] == periods[k] and t - before <= times[j] <= t + after
        ]
        out.append(float(np.exp(np.mean(window))))
    return out


@pytest.mark.parametrize("before, after", [(1, 1), (6, 0), (0.5, 0.5)])
def test_smoothing_averages_measured_readings_within_the_window_and_the_period(before, after):
    rows = rows_of(READINGS)
    got = smoothed(rows, before, after)
    expected = np.repeat(brute_force(READINGS, before, after), 2)
    np.testing.assert_allclose(got["turbulence_intensity"].to_numpy(), expected, rtol=1e-12)
    assert got.drop("turbulence_intensity").equals(rows.drop("turbulence_intensity"))


def test_without_a_window_the_readings_stay_exactly_as_they_were():
    rows = rows_of(READINGS)
    assert smoothed(rows, 0, 0).equals(rows)


def test_smoothing_refits_windfold_on_each_variant_and_keeps_what_is_unchanged(
    tiny_study, tmp_path
):
    work = tmp_path / "work"
    options = {"farms": ("kelmarsh",), "configs": tiny_study.configs, "variants": ("h00", "h03")}
    with pytest.raises(ValueError, match="missing"):
        inflow_smoothing(tiny_study.data, tiny_study.runs, work, **options)
    result = inflow_smoothing(tiny_study.data, tiny_study.runs, work, refit=True, **options)
    farm = result["kelmarsh"]
    assert farm["splits"] == 2 and farm["raw_refits_identical"] == 2
    assert farm["minus_raw"]["h00"] == {"estimate": 0.0, "low": 0.0, "high": 0.0}
    assert farm["skill"]["h00"] == farm["skill"]["windfold"]
    assert farm["minus_raw"]["h03"]["estimate"] == pytest.approx(
        farm["skill"]["h03"] - farm["skill"]["windfold"]
    )
    assert "large_farms_minus_raw" not in result

    source = tiny_study.data / "prepared" / "kelmarsh"
    prepared = work / "data" / "h03" / "prepared" / "kelmarsh"
    study_table = pl.read_parquet(source / inflow.TABLE)
    table = pl.read_parquet(prepared / inflow.TABLE)
    assert table.select("timestamp", "turbine").equals(study_table.select("timestamp", "turbine"))
    unmeasured = ~inflow.measured(pl.read_parquet(source / "rows.parquet"))
    assert (table["ratio"].to_numpy()[unmeasured] == 1.0).all()
    assert not np.allclose(table["ratio"].to_numpy(), study_table["ratio"].to_numpy())
    assert (prepared / "kernel_design").resolve() == (source / "kernel_design").resolve()
    assert inflow_smoothing(tiny_study.data, tiny_study.runs, work, **options) == result

    folders = (tiny_study.data, tiny_study.runs, work)
    changed = shutil.copytree(tiny_study.configs, tmp_path / "configs")
    protocol = changed / "protocol.toml"
    protocol.write_text(protocol.read_text().replace("trials = 2", "trials = 3"))
    with pytest.raises(
        ValueError, match="h00/prepared/kelmarsh/inflow_physics.parquet was made by"
    ):
        inflow_smoothing(*folders, refit=True, **(options | {"configs": changed}))
    limited = shutil.copytree(tiny_study.configs, tmp_path / "limited")
    study = limited / "studies" / "development.toml"
    study.write_text(study.read_text() + "\n[limits]\ndays = 1\n")
    with pytest.raises(ValueError, match="windfold/arc_090_single.json was made by other code"):
        inflow_smoothing(*folders, refit=True, **(options | {"configs": limited}))
    fits = work / "fits" / "h03" / "development" / "predictions" / "kelmarsh" / "windfold"
    fitted = json.loads((fits / "arc_270_single.json").read_text())
    del fitted["provenance"]
    (fits / "arc_270_single.json").write_text(json.dumps(fitted))
    with pytest.raises(ValueError, match="arc_270_single.json has no provenance; refit it under"):
        inflow_smoothing(*folders, refit=True, **options)
    (work / "records" / "kelmarsh" / "h03.json").unlink()
    with pytest.raises(ValueError, match="h03/prepared/kelmarsh/inflow_physics.parquet has no"):
        inflow_smoothing(*folders, refit=True, **options)
