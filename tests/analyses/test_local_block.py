import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.analyses.development import prediction_file, record_file
from windfold.analyses.local_block import local_block

SPLITS = ("arc_000_single", "arc_030_single", "arc_060_single")
WAKED_ROWS = (1, 3, 2)
MEANS = {
    "windfold": (0.5, 0.8, 0.0),
    "windfold_no_local": (0.6, 0.5, 0.2),
    "pywake_inflow": (0.4, 0.9, 0.1),
    "windfold_unit_physics": (0.3, 0.3, 0.3),
    "no_wake": (0.0, 0.0, 0.0),
}
FACTS = {
    "windfold": {"physics": (0.9, 1.1, 0.7), "kernel": ([3.0, 4.0], [1.0, 1.0], [0.0, 2.0])},
    "windfold_no_local": {
        "physics": (1.0, 1.2, 1.4),
        "kernel": ([2.0, 2.0], [0.0, 0.0], [6.0, 8.0]),
    },
}


def write_study(runs) -> None:
    """three single arcs whose waked test rows have y = 1 and a constant prediction per model
    and arc, each beside one wild unwaked row; a fourth included arc has no waked test rows."""
    folder = runs / "development" / "manifests"
    folder.mkdir(parents=True)
    pl.DataFrame(
        {
            "split": [*SPLITS, "arc_090_single", "arc_120_single"],
            "regime": ["arc_single"] * 5,
            "status": ["included"] * 4 + ["excluded"],
            "test_waked_rows": [*WAKED_ROWS, 0, 4],
        }
    ).write_csv(folder / "kelmarsh_summary.csv")
    start = datetime(2019, 3, 1, tzinfo=UTC)
    for k, (split, waked) in enumerate(zip(SPLITS, WAKED_ROWS, strict=True)):
        keys = pl.DataFrame(
            {
                "timestamp": [start + timedelta(days=k, minutes=10 * r) for r in range(waked + 1)],
                "turbine": ["T1"] * (waked + 1),
                "y": [1.0] * waked + [5.0],
                "waked": [True] * waked + [False],
            }
        )
        for model, means in MEANS.items():
            path = prediction_file(runs, "kelmarsh", model, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            keys.with_columns(mean=pl.Series([means[k]] * waked + [-5.0])).write_parquet(path)
        for model, facts in FACTS.items():
            blocks = {"physics": [facts["physics"][k]], "shared_kernel": facts["kernel"][k]}
            record_file(runs, "kelmarsh", model, split).write_text(
                json.dumps({"facts": {"blocks": blocks}})
            )


def test_local_block_signs_the_arc_gaps_and_weighs_residuals_by_row(tmp_path):
    write_study(tmp_path)
    got = local_block(tmp_path, ("kelmarsh",))
    assert list(got) == ["kelmarsh"]
    farm = got["kelmarsh"]
    skill = {model: [100 * (1 - (1 - c) ** 2) for c in means] for model, means in MEANS.items()}
    local = [a - b for a, b in zip(skill["windfold"], skill["windfold_no_local"], strict=True)]
    physics = [a - b for a, b in zip(skill["windfold"], skill["pywake_inflow"], strict=True)]
    assert local == pytest.approx([-9.0, 21.0, -36.0])
    assert physics == pytest.approx([11.0, -3.0, -19.0])
    assert farm["per_split"] == {
        "split": list(SPLITS),
        "windfold_minus_no_local": pytest.approx(local),
        "windfold_minus_pywake_inflow": pytest.approx(physics),
        "beta": [0.9, 1.1, 0.7],
    }
    assert (farm["splits"], farm["splits_removing_local_helps"]) == (3, 2)
    assert farm["splits_windfold_beats_pywake_inflow"] == 1
    assert farm["beta_against_gap_to_pywake_inflow"] == pytest.approx({"spearman": 0.5, "p": 2 / 3})
    assert (farm["beta_median"], farm["beta_median_without_local"]) == (0.9, 1.2)
    assert farm["kernel_rms_median"] == pytest.approx(2**0.5)
    assert farm["kernel_rms_median_without_local"] == pytest.approx(2.0)
    rows = sum(WAKED_ROWS)
    assert farm["mean_waked_residual"] == pytest.approx(
        {
            model: sum(n * (1 - c) for n, c in zip(WAKED_ROWS, means, strict=True)) / rows
            for model, means in MEANS.items()
        }
    )
