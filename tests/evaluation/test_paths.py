from windfold.config import load_protocol
from windfold.evaluation import paths

SPLIT = "arc_000_single"


def source(sensitivity: str, model: str) -> str:
    table, _ = paths.fit_inputs(
        "development", "kelmarsh", sensitivity, model, SPLIT, None, load_protocol()
    )
    return table.removesuffix(f"/predictions/kelmarsh/{model}/{SPLIT}.parquet")


def test_block_length_sensitivities_read_the_main_reductions():
    protocol = load_protocol()
    assert paths.reduction_source(protocol, "block_days_14") is None
    assert paths.reduction_source(protocol, "kernel_along_20d") == "kernel_along_20d"
    assert paths.reduction_source(protocol, None) is None


def test_refitting_sensitivities_read_the_models_they_do_not_refit_from_the_main_study():
    own = "development/sensitivities/{}"
    assert source("kernel_along_20d", "pywake") == "development"
    assert source("kernel_along_20d", "windfold") == own.format("kernel_along_20d")
    assert source("fixed_shrinkage", "no_wake") == "development"
    assert source("fixed_shrinkage", "kernel_linear") == own.format("fixed_shrinkage")
    assert source("fixed_shrinkage", "windfold") == "development"
    assert source("arc_buffer_0deg", "pywake") == own.format("arc_buffer_0deg")


def test_a_split_sensitivity_borrows_the_regimes_it_does_not_refit():
    protocol = load_protocol()
    assert paths.borrowed_regimes(protocol, "arc_buffer_0deg") == (
        "random",
        "time_blocked",
        "history",
        "threshold",
    )
    assert paths.borrowed_regimes(protocol, "fixed_shrinkage") == ()
    assert paths.borrowed_regimes(protocol, "speed_4_to_12") == ()
