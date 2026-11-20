import polars as pl
import pytest

from windfold.analyses.practical_scale import practical_scale


def pooled(errors: pl.DataFrame, model: str, arcs, quantity: str = "sse") -> float:
    """a model's per-row quantity over the arcs' waked rows, pooled per seed, then averaged."""
    part = errors.filter(
        pl.col("model") == model, pl.col("split").is_in(list(arcs)), pl.col("waked")
    )
    per_seed = part.group_by("seed").agg(mean=pl.col(quantity).sum() / pl.col("rows").sum())
    return float(per_seed["mean"].to_numpy().mean())


def test_windfold_meets_physics_and_the_best_learned_model_on_the_planned_bootstrap(
    confirmatory_study,
):
    with pytest.raises(FileNotFoundError, match="second freeze"):
        practical_scale(confirmatory_study.runs, "confirmatory")
    (confirmatory_study.runs / "confirmatory" / "freeze_second.json").write_text("{}")
    farm = practical_scale(confirmatory_study.runs, "confirmatory")["penmanshiel"]
    assert farm["strongest_learned"] == "identity_gbm_inflow"
    assert farm["accuracy"]["windfold"]["mae_kw"] == pytest.approx(
        {"estimate": 70.0, "low": 63.0, "high": 77.0}
    )
    errors = confirmatory_study.errors
    for other in ("pywake_inflow", "identity_gbm_inflow"):
        theirs, own, base = (
            pooled(errors, m, confirmatory_study.arcs) for m in (other, "windfold", "no_wake")
        )
        kw = (
            pooled(errors, m, confirmatory_study.arcs, "abs_error_kw") for m in (other, "windfold")
        )
        got = {name: value["estimate"] for name, value in farm["against"][other].items()}
        assert got == pytest.approx(
            {
                "mse_reduction": theirs - own,
                "relative_mse_reduction_percent": 100 * (1 - own / theirs),
                "skill_points": 100 * (theirs - own) / base,
                "mae_reduction_kw": next(kw) - next(kw),
            }
        )
        skill = farm["against"][other]["skill_points"]
        assert skill["low"] < skill["estimate"] < skill["high"] and skill["p"] < 0.05
    first = confirmatory_study.arcs[0]
    waked = errors.filter(pl.col("model") == "windfold", pl.col("split") == first, pl.col("waked"))
    assert [arc["split"] for arc in farm["per_arc"]] == list(confirmatory_study.arcs[:2])
    assert farm["per_arc"][0]["waked_rows"] == waked["rows"].sum()
    theirs, own, base = (
        pooled(errors, m, [first]) for m in ("pywake_inflow", "windfold", "no_wake")
    )
    skill = farm["per_arc"][0]["against_pywake_inflow"]["skill_points"]["estimate"]
    assert skill == pytest.approx(100 * (theirs - own) / base)
    families, models = farm["cost"]["families"], farm["cost"]["models"]
    assert families["engine"] == {
        "jobs": 3,
        "gpu_jobs": 0,
        "seconds": {"median": 201.0, "max": 202.0},
        "peak_memory_gb": {"median": 2.01, "max": 2.02},
    }
    assert (families["gbm"]["jobs"], families["graph"]["gpu_jobs"]) == (9, 1)
    assert models["windfold"]["fit_seconds"] == {"median": 11.0, "max": 12.0}
    assert (models["identity_gbm_inflow"]["fits"], models["pywake_inflow"]["threads"]) == (9, [1])
