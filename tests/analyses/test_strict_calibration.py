import json
import shutil

import numpy as np
import polars as pl
import pytest

from windfold.analyses.strict_calibration import arc_start, held_out, strict_calibration


def test_the_held_out_arc_takes_its_buffers_on_both_sides_around_north():
    directions = np.array([324.9, 325.0, 359.9, 0.0, 4.9, 5.0, 180.0])
    assert held_out(directions, 330.0, 30.0, 5.0).tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    directions = np.array([354.9, 355.0, 0.0, 34.9, 35.0])
    assert held_out(directions, 0.0, 30.0, 5.0).tolist() == [False, True, True, True, False]


def test_only_single_arcs_name_an_arc_to_hold_out():
    assert arc_start("arc_030_single") == 30.0
    with pytest.raises(ValueError, match="not a single-arc split"):
        arc_start("arc_030_opposite")


def waked_sse(root, model, split):
    path = root / "development" / "predictions" / "kelmarsh" / model / f"{split}.parquet"
    frame = pl.read_parquet(path).filter(pl.col("waked"))
    return float(((frame["y"] - frame["mean"]) ** 2).sum())


def test_each_arc_is_recalibrated_without_its_calibration_rows_and_kept_while_unchanged(
    tiny_study, tmp_path
):
    work = tmp_path / "work"
    farms = ("kelmarsh",)
    folders = (tiny_study.data, tiny_study.runs, work, farms)
    with pytest.raises(ValueError, match="not recalibrated"):
        strict_calibration(*folders, configs=tiny_study.configs)
    made = strict_calibration(*folders, refit=True, configs=tiny_study.configs)
    result = made["kelmarsh"]

    rows = pl.read_parquet(tiny_study.data / "prepared" / "kelmarsh" / "rows.parquet")
    calibration = rows.filter(pl.col("period") == "calibration")["farm_direction"].to_numpy()
    excluded = []
    for split, (low, high) in {"arc_090_single": (85, 125), "arc_270_single": (265, 305)}.items():
        record = json.loads((work / "records" / "kelmarsh" / f"{split}.json").read_text())
        kept = int(((calibration < low) | (calibration >= high)).sum())
        assert (record["kept_calibration_rows"], record["calibration_rows"]) == (
            kept,
            calibration.size,
        )
        assert record["parameters"] != record["shared_parameters"]
        excluded.append(1 - kept / calibration.size)
        prepared = work / "data" / split / "prepared" / "kelmarsh"
        assert (prepared / "rows.parquet").resolve() == (
            tiny_study.data / "prepared" / "kelmarsh" / "rows.parquet"
        ).resolve()
        table = prepared / "inflow_physics.parquet"
        assert not table.is_symlink()
        assert (
            pl.read_parquet(table).height == rows.filter(pl.col("period") == "development").height
        )
    assert result["splits"] == 2
    assert result["excluded_calibration_share"] == pytest.approx(np.mean(excluded))

    splits = tiny_study.splits
    reference = sum(waked_sse(tiny_study.runs, "no_wake", s) for s in splits)
    for model in ("windfold", "pywake_inflow"):
        shared = 100 * (1 - sum(waked_sse(tiny_study.runs, model, s) for s in splits) / reference)
        refits = sum(waked_sse(work / "fits" / s, model, s) for s in splits)
        strict = 100 * (1 - refits / reference)
        change = result[model]["strict_minus_shared"]
        assert result[model]["shared"] == pytest.approx(shared)
        assert result[model]["strict"] == pytest.approx(strict)
        assert change["estimate"] == pytest.approx(strict - shared)
        assert change["low"] <= change["estimate"] <= change["high"]

    assert strict_calibration(*folders, configs=tiny_study.configs) == made
    changed = shutil.copytree(tiny_study.configs, tmp_path / "configs")
    protocol = changed / "protocol.toml"
    protocol.write_text(protocol.read_text().replace("trials = 2", "trials = 3"))
    with pytest.raises(ValueError, match="made by other code, settings or inputs; refit it under"):
        strict_calibration(*folders, refit=True, configs=changed)
    limited = shutil.copytree(tiny_study.configs, tmp_path / "limited")
    study = limited / "studies" / "development.toml"
    study.write_text(study.read_text() + "\n[limits]\ndays = 1\n")
    with pytest.raises(ValueError, match="arc_090_single.json was made by other code, settings"):
        strict_calibration(*folders, refit=True, configs=limited)
    record = work / "records" / "kelmarsh" / "arc_270_single.json"
    kept = json.loads(record.read_text())
    del kept["provenance"]
    record.write_text(json.dumps(kept))
    with pytest.raises(ValueError, match="arc_270_single.json has no provenance; refit it under"):
        strict_calibration(*folders, refit=True, configs=tiny_study.configs)
