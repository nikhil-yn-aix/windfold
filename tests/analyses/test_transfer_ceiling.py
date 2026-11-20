from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from windfold.analyses.transfer_ceiling import (
    FOLDS,
    Arc,
    cross_fitted,
    fit_arc,
    load_farm,
    nested,
    score_left_out,
    shared_terms,
    transfer_ceiling,
)


def test_each_fold_is_predicted_by_the_better_penalty_fitted_on_the_other_folds():
    rng = np.random.default_rng(3)
    rows = 300
    x = rng.normal(size=(rows, 3))
    fold = np.arange(rows) % FOLDS
    waked = rng.random(rows) < 0.6
    arc = Arc(x, np.zeros(rows), np.zeros(rows), waked, fold, np.zeros(waked.sum(), int), 1)
    residual = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, rows)
    loose, tight, columns = 1e-3 * np.eye(3), 1e4 * np.eye(3), np.arange(3)
    expected = np.zeros(rows)
    for k in range(FOLDS):
        other = fold != k
        theta = np.linalg.solve(x[other].T @ x[other] + loose, x[other].T @ residual[other])
        expected[fold == k] = x[fold == k] @ theta
    np.testing.assert_allclose(cross_fitted(arc, residual, columns, [tight, loose]), expected)
    moved = residual.copy()
    moved[fold == 0] += 100.0
    again = cross_fitted(arc, moved, columns, [loose])
    np.testing.assert_allclose(again[fold == 0], expected[fold == 0])


def own_fit_errors(farm, arcs):
    """arcs by penalties: each arc's waked-row error of the shared terms fitted on its own
    training buckets."""
    table = []
    for arc in arcs:
        fits, family, target = shared_terms(farm, arc.base, ~arc.base.held_out)
        table.append([arc.waked.error(farm.columns(family), theta, target) for theta in fits])
    return np.array(table)


def test_an_arcs_nested_penalty_ignores_its_own_rows_where_the_score_left_out_one_does_not(
    tiny_study,
):
    farm = load_farm(tiny_study.data, "kelmarsh", tiny_study.configs)
    arcs = [fit_arc(farm, tiny_study.runs, s, tiny_study.configs) for s in tiny_study.splits]
    cross = farm.cross.copy()
    cross[arcs[0].base.held_out] *= 10.0
    moved = replace(farm, cross=cross)
    assert nested(moved, arcs, shared_terms)[0] == nested(farm, arcs, shared_terms)[0]
    before, after = (score_left_out(own_fit_errors(f, arcs))[0] for f in (farm, moved))
    assert before != after


def waked_sse(path):
    frame = pl.read_parquet(path).filter(pl.col("waked"))
    return float(((frame["y"] - frame["mean"]) ** 2).sum())


def test_points_pool_the_arcs_waked_errors_over_no_wakes(tiny_study):
    result = transfer_ceiling(tiny_study.data, tiny_study.runs, ("kelmarsh",), tiny_study.configs)
    farm = result["farms"]["kelmarsh"]
    study = tiny_study.runs / "development"
    folders = {
        model: study / "predictions" / "kelmarsh" / model
        for model in ("no_wake", "pywake_inflow", "windfold", "windfold_no_local")
    }
    folders["restricted"] = study / "secondary" / "windfold_restricted" / "predictions" / "kelmarsh"
    sse = {
        model: sum(waked_sse(folder / f"{split}.parquet") for split in tiny_study.splits)
        for model, folder in folders.items()
    }
    waked = [
        pl.read_parquet(folders["no_wake"] / f"{s}.parquet")["waked"] for s in tiny_study.splits
    ]
    assert (farm["arcs"], farm["waked_rows"]) == (2, sum(w.sum() for w in waked))
    points = {name: value["estimate"] for name, value in farm["points"].items()}
    physics, base = sse["pywake_inflow"], sse["no_wake"]
    assert points["pywake_inflow_error"] == pytest.approx(100 * physics / base)
    assert points["windfold"] == pytest.approx(100 * (physics - sse["windfold"]) / base)
    assert points["windfold_no_local"] == pytest.approx(
        100 * (physics - sse["windfold_no_local"]) / base
    )
    restricted = 100 * (sse["windfold"] - sse["restricted"]) / base
    assert points["windfold_restricted_minus_windfold"] == pytest.approx(restricted)
