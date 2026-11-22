import shutil
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from windfold import config
from windfold.cli import app
from windfold.config import (
    ConfigError,
    apply_overrides,
    check_model_names,
    load_farm,
    load_models,
    load_protocol,
    load_resources,
    load_study,
    resolve_models,
    tuned_model,
    with_sensitivity,
)

FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie", "smarteole")


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    source = Path(str(resources.files("windfold") / "configs"))
    target = tmp_path / "configs"
    shutil.copytree(source, target)
    return target


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


def test_packaged_configs_load_and_agree():
    protocol = load_protocol()
    models = load_models()
    load_resources()
    for farm in FARMS:
        assert load_farm(farm).name == farm
    for name in ("smoke", "development", "confirmatory"):
        study = load_study(name)
        assert set(study.models) <= set(models)
        assert set(study.sensitivities) <= set(protocol.sensitivities)
    check_model_names(protocol, models)


def test_the_configs_command_checks_turbines_menus_and_the_report(configs: Path, monkeypatch):
    monkeypatch.setattr(config, "resources", SimpleNamespace(files=lambda package: configs.parent))
    assert CliRunner().invoke(app, ["configs"]).exit_code == 0
    for name, old, new in (
        ("turbines/senvion_mm92.toml", "rotor_diameter_m = 92.5", 'rotor_diameter_m = "wide"'),
        ("menus/graph.toml", "index = 1\n", "index = 2\n"),
        ("report.toml", "raster_dpi = 300", "raster_dpi = 0"),
    ):
        text = (configs / name).read_text()
        edit(configs / name, old, new)
        checked = CliRunner().invoke(app, ["configs"])
        assert isinstance(checked.exception, ConfigError) and name in str(checked.exception)
        (configs / name).write_text(text)


def test_run_matrix_is_scoped(configs: Path):
    edit(configs / "protocol.toml", '"pywake", "direction_lookup"]', '"pywake", "lookup"]')
    with pytest.raises(ConfigError, match="monitoring.models: unknown models"):
        check_model_names(load_protocol(configs), load_models(configs))
    head = '[sensitivities.kernel_across_2d]\noverrides = { "kernel.across_max_d" = 2.0 }\n'
    edit(configs / "protocol.toml", head + 'models = ["kernel_linear", "windfold"]\n', head)
    with pytest.raises(ConfigError, match="names both the models and the regimes"):
        load_protocol(configs)
    edit(configs / "models.toml", 'regimes = ["arc_single", "history"]', 'regimes = ["threshold"]')
    with pytest.raises(ConfigError, match="threshold fits come from monitoring.models"):
        load_models(configs)


def test_unknown_key_is_named(configs: Path):
    edit(configs / "protocol.toml", "stuck_repeats = 3", "stuck_repeats = 3\nstuck_repeat = 3")
    with pytest.raises(ConfigError, match="processing.stuck_repeat"):
        load_protocol(configs)


def test_wrong_type_is_named(configs: Path):
    edit(configs / "protocol.toml", "random_folds = 5", 'random_folds = "five"')
    with pytest.raises(ConfigError, match="splits.random_folds"):
        load_protocol(configs)


def test_overlapping_windows_fail(configs: Path):
    edit(
        configs / "farms" / "penmanshiel.toml",
        "confirmatory = { start = 2020-01-01T00:00:00Z",
        "confirmatory = { start = 2019-06-01T00:00:00Z",
    )
    with pytest.raises(ConfigError, match="overlap"):
        load_farm("penmanshiel", configs)


def test_window_ending_before_start_fails(configs: Path):
    edit(
        configs / "farms" / "smarteole.toml",
        "end = 2020-05-25T00:00:00Z }\n\n[histories]",
        "end = 2020-01-01T00:00:00Z }\n\n[histories]",
    )
    with pytest.raises(ConfigError, match="not after its start"):
        load_farm("smarteole", configs)


def test_naive_timestamps_fail(configs: Path):
    edit(
        configs / "farms" / "smarteole.toml",
        "starts = [2020-02-17T16:30:00Z]",
        "starts = [2020-02-17T16:30:00]",
    )
    with pytest.raises(ConfigError, match="histories.starts"):
        load_farm("smarteole", configs)


def model(**fields) -> dict:
    return {"display_name": "x", "representation": "identity", **fields}


def test_unknown_block_fails():
    with pytest.raises(ConfigError, match="unknown blocks"):
        resolve_models({"a": model(blocks=["turbine_offset", "wake_magic"])})


def test_inheritance_cycle_fails():
    raw = {"a": model(base="b", blocks=["turbine_offset"]), "b": model(base="a")}
    with pytest.raises(ConfigError, match="cycle"):
        resolve_models(raw)


def test_unknown_base_fails():
    with pytest.raises(ConfigError, match="unknown model"):
        resolve_models({"a": model(base="missing")})


def test_dropping_a_missing_block_fails():
    raw = {"a": model(blocks=["turbine_offset"]), "b": model(base="a", drop=["physics"])}
    with pytest.raises(ConfigError, match="drops"):
        resolve_models(raw)


def test_permuted_layout_needs_shared_kernel():
    raw = {"a": model(blocks=["turbine_offset"], kernel_layout="permuted")}
    with pytest.raises(ConfigError, match="shared_kernel"):
        resolve_models(raw)


def test_child_inherits_and_drops():
    raw = {
        "full": model(blocks=["turbine_offset", "physics", "local"], physics_scale="learned"),
        "lean": {"base": "full", "display_name": "lean", "drop": ["physics"]},
    }
    lean = resolve_models(raw)["lean"]
    assert lean.blocks == ("turbine_offset", "local")
    assert lean.physics_scale is None
    assert lean.representation == "identity"


def test_child_must_name_itself_for_display():
    raw = {"full": model(blocks=["turbine_offset"]), "copy": {"base": "full"}}
    with pytest.raises(ConfigError, match="display_name"):
        resolve_models(raw)


def test_inflow_and_scales_stay_out_of_a_spec_text_at_their_defaults():
    raw = {
        "plain": model(blocks=["turbine_offset", "physics"], physics_scale="learned"),
        "measured": model(
            blocks=["turbine_offset", "physics"],
            physics_scale="learned",
            inflow="measured",
            scales="fixed",
        ),
    }
    specs = resolve_models(raw)
    plain, measured = str(specs["plain"]), str(specs["measured"])
    assert "inflow" not in plain and "scales" not in plain
    assert "inflow='measured'" in measured and "scales='fixed'" in measured
    assert plain.replace("plain", "measured") == measured.replace(
        " inflow='measured' scales='fixed'", ""
    )


def test_measured_inflow_needs_physics_or_a_gbm_or_graph_learner():
    for entry in (
        model(blocks=["turbine_offset", "shared_kernel"], inflow="measured"),
        model(learner="automl", features=["peer_power"], inflow="measured"),
        model(learner="graph", edges="wind_frame", pretrained=True, inflow="measured"),
    ):
        with pytest.raises(ConfigError, match="measured inflow"):
            resolve_models({"a": entry})
    for entry in (
        model(learner="gbm", features=["reference_speed"], inflow="measured"),
        model(learner="graph", edges="absolute", inflow="measured"),
    ):
        assert resolve_models({"a": entry})["a"].inflow == "measured"


def test_an_inflow_control_needs_the_physics_block():
    for entry in (
        model(blocks=["turbine_offset", "shared_kernel"], inflow="diurnal"),
        model(learner="gbm", features=["reference_speed"], inflow="average"),
    ):
        with pytest.raises(ConfigError, match="diurnal|average"):
            resolve_models({"a": entry})
    blocks = ["turbine_offset", "physics"]
    spec = model(blocks=blocks, physics_scale="learned", inflow="climatology")
    assert resolve_models({"a": spec})["a"].inflow == "climatology"


def test_scales_belong_to_engine_models():
    with pytest.raises(ConfigError, match="scales"):
        resolve_models({"a": model(learner="gbm", features=["turbine"], scales="fixed")})


def test_a_child_of_a_learner_with_measured_inflow_keeps_it():
    raw = {
        "parent": model(learner="gbm", features=["reference_speed"], inflow="measured"),
        "child": {"base": "parent", "display_name": "child"},
    }
    assert resolve_models(raw)["child"].inflow == "measured"


def test_dropping_physics_drops_measured_inflow():
    raw = {
        "full": model(
            blocks=["turbine_offset", "physics", "shared_kernel"],
            physics_scale="learned",
            inflow="measured",
            scales="fixed",
        ),
        "lean": {"base": "full", "display_name": "lean", "drop": ["physics"]},
    }
    lean = resolve_models(raw)["lean"]
    assert lean.inflow == "constant" and lean.scales == "fixed"


def test_a_pretrained_graph_tunes_like_the_graph_with_its_edges_and_inflow():
    specs = load_models()
    assert tuned_model("pretrained_graph", specs) == "wind_frame_graph"
    assert tuned_model("wind_frame_graph_inflow", specs) == "wind_frame_graph_inflow"


def test_sensitivity_changes_only_its_setting():
    protocol = load_protocol()
    farms = {name: load_farm(name) for name in FARMS}
    changed, changed_farms = with_sensitivity("arcs_60deg", protocol, farms)
    assert (changed.splits.arc_width_deg, changed.splits.arc_count) == (60.0, 6)
    assert (
        changed.splits.model_copy(update={"arc_width_deg": 30.0, "arc_count": 12})
        == protocol.splits
    )
    assert changed.processing == protocol.processing
    assert changed_farms == farms


def test_farm_sensitivity_moves_one_window():
    protocol = load_protocol()
    farms = {name: load_farm(name) for name in FARMS}
    _, changed = with_sensitivity("hill_of_towie_after_tune_up", protocol, farms)
    start = changed["hill_of_towie"].windows.development.start
    assert (start.year, start.month, start.day, start.hour) == (2024, 3, 14, 17)
    assert (
        changed["hill_of_towie"].windows.development.end
        == farms["hill_of_towie"].windows.development.end
    )
    assert changed["kelmarsh"] == farms["kelmarsh"]


def test_override_of_a_missing_setting_fails():
    with pytest.raises(ConfigError, match="names no existing setting"):
        apply_overrides({"splits": {"arc_count": 12}}, {"splits.arc_counts": 6}, "test")
