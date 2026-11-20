"""a stand-in cpu load: the same kind of work as a production fit, repeated until it is stopped.

while a full_load job runs, an automl budget fit, the runner puts one across every cpu slot with
no job and on every idle core of a slot holding one-core jobs, so the budget is chosen with every
core busy, as when production automl fits, which get no load jobs, run beside the other fits.
the runner kills them when the last full_load job ends. nothing is written.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from windfold.runner.worker import allotment

ROUNDS_PER_FIT = 200


def fit_forever(rows: int, features: int, seed: int, out_dir: Path) -> None:
    threads, _ = allotment()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, features))
    y = x @ rng.standard_normal(features) + rng.standard_normal(rows)
    data = lgb.Dataset(x, label=y, params={"verbose": -1})
    params = {
        "objective": "regression",
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
        "verbose": -1,
    }
    while True:
        lgb.train(params, data, num_boost_round=ROUNDS_PER_FIT)
