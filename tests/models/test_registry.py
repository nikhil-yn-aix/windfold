import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from scipy.interpolate import BSpline

from windfold import geometry, inflow
from windfold.config import load_models, load_protocol
from windfold.models import blocks, engine, registry
from windfold.models.interface import FitContext
from windfold.schemas import LAYOUT, PREPARED_ROWS, conform

DIAMETER = 82.0
EAST = np.array([0.0, 410.0, 820.0, 200.0, 610.0])
NORTH = np.array([0.0, 20.0, -30.0, 400.0, 420.0])
BASIS = blocks.kernel_basis(30.0, 3.0, 1.5, 9, 0.45, 4, 3, 3)
SETTINGS = registry.EngineSettings(
    speed_bins=3,
    buckets=72,
    harmonics_order=4,
    spline_knots=24,
    spline_ridge=0.1,
    offset_variance=1.0,
    physics_variance=0.25,
    starts=(1e-4, 1e-2, 1.0),
    bounds=engine.Bounds(1e-6, 1e-8, 100.0),
    fixed_prior_variance=None,
    lookup_window_deg=5,
    kernel=BASIS,
    kernel_ridge=0.1,
)


def wake(along, across):
    return -0.4 * np.exp(-0.5 * (across / 0.6) ** 2) * np.exp(-along / 8.0) * (along > 0)


def prepared_rows(timestamps: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(EAST)
    full = (1 << n) - 1
    masks = np.where(
        rng.random(timestamps) < 0.2, full & ~(1 << rng.integers(0, n, timestamps)), full
    )
    direction = np.mod(np.cumsum(rng.normal(0, 8, timestamps)) + 250, 360)
    speed_bin = rng.integers(0, 3, timestamps)
    t, i = np.nonzero((masks[:, None] >> np.arange(n)) & 1)
    along, across = geometry.relations(EAST, NORTH, direction[t], DIAMETER)
    operating = ((masks[t][:, None] >> np.arange(n)) & 1).astype(bool)
    source = geometry.upstream(along[np.arange(len(t)), i], across[np.arange(len(t)), i], 30.0, 3.0)
    effect = (
        wake(along[np.arange(len(t)), i], across[np.arange(len(t)), i]) * (source & operating)
    ).sum(1)
    raw = np.array([0.02, -0.01, 0.0, 0.03, -0.04])[i] + effect + rng.normal(0, 0.03, len(t))
    y = engine.centre(raw, t.astype(np.int64))
    count = np.bincount(t)[t]
    stamps = np.datetime64("2018-01-01T00:00", "us") + t * np.timedelta64(10, "m")
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series(stamps).dt.replace_time_zone("UTC"),
            "turbine": [f"T{k}" for k in i],
            "turbine_index": pl.Series(i, dtype=pl.UInt8),
            "period": ["development"] * len(t),
            "power_kw": 800.0 * np.exp(y),
            "speed": 7.0 + rng.normal(0, 0.3, len(t)),
            "speed_std": np.full(len(t), 0.7),
            "direction_raw": direction[t],
            "direction": direction[t],
            "nacelle_position": direction[t],
            "farm_direction": direction[t],
            "reference_speed": np.full(len(t), 7.0),
            "speed_bin": pl.Series(speed_bin[t], dtype=pl.UInt8),
            "direction_bucket": pl.Series((direction[t] // 5).astype(int), dtype=pl.UInt8),
            "steady": np.ones(len(t), bool),
            "operating_mask": pl.Series(masks[t], dtype=pl.UInt32),
            "waked": source.any(1),
            "valid_turbines": pl.Series(count, dtype=pl.UInt8),
            "y": y,
            "temperature": np.full(len(t), 10.0),
            "turbulence_intensity": np.full(len(t), 0.1),
        }
    )
    return conform(frame, PREPARED_ROWS)


@pytest.fixture(scope="module")
def farm(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("prepared")
    rows = prepared_rows(150, seed=1)
    rows.write_parquet(folder / "rows.parquet")
    layout = pl.DataFrame(
        {
            "turbine": [f"T{k}" for k in range(len(EAST))],
            "latitude": np.zeros(len(EAST)),
            "longitude": np.zeros(len(EAST)),
            "x_m": EAST,
            "y_m": NORTH,
            "hub_height_m": np.full(len(EAST), 59.0),
        }
    )
    conform(layout, LAYOUT).write_parquet(folder / "layout.parquet")
    neighbours = []
    for row in rows.iter_rows(named=True):
        along, across = geometry.relations(EAST, NORTH, row["farm_direction"], DIAMETER)
        near = geometry.upstream(along[row["turbine_index"]], across[row["turbine_index"]], 30, 3)
        neighbours.append(int(near.sum()))
    count = np.array(neighbours, dtype=np.float64)
    physics = rows.select("timestamp", "turbine").with_columns(
        physics_jensen=-0.10 * count,
        physics_niayifar=-0.25 * count,
        physics_turbopark=-0.60 * count,
        physics_jensen_default=-0.05 * count,
        physics_niayifar_default=-0.15 * count,
        physics_turbopark_default=-0.30 * count,
    )
    physics.write_parquet(folder / "physics.parquet")
    rng = np.random.default_rng(11)
    stamps = rows["timestamp"].unique().sort()
    ratio = pl.DataFrame({"timestamp": stamps, "ratio": np.exp(rng.normal(0.0, 0.3, len(stamps)))})
    measured = physics.join(ratio, on="timestamp", how="left", maintain_order="left")
    measured = measured.with_columns(
        [(pl.col(c) * pl.col("ratio") ** -0.5).alias(c) for c in physics.columns[2:]]
    )
    for scale, name in enumerate(inflow.TABLES.values(), start=1):
        changed = measured.with_columns(pl.col("^physics_.*$") * (1 + 0.1 * scale))
        conform(changed, inflow.INFLOW_PHYSICS).write_parquet(folder / name)
    blocks.build_kernel_designs(folder, folder / "kernel_design", DIAMETER, BASIS, 30.0, 3.0, 7)
    return folder


def split(folder: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = pl.read_parquet(folder / "rows.parquet")
    stamps = rows["timestamp"].unique().sort()
    cut = stamps[int(0.8 * len(stamps))]
    return rows.filter(pl.col("timestamp") < cut), rows.filter(pl.col("timestamp") >= cut)


def context(folder: Path) -> FitContext:
    return FitContext("test", folder, 0, 1, None)


@pytest.fixture(scope="module")
def windfold(farm) -> registry.FittedEngine:
    """windfold fitted once on the farm's training rows, for the tests that only read the fit."""
    train, _ = split(farm)
    fitted = registry.model_from_blocks(load_models()["windfold"], SETTINGS).fit(
        train, context(farm)
    )
    assert isinstance(fitted, registry.FittedEngine)
    return fitted


def test_every_blocks_model_fits_and_predicts_centred_rows(farm, windfold):
    train, test = split(farm)
    specs = [s for s in load_models().values() if s.blocks]
    assert {s.name for s in specs} >= {
        "no_wake",
        "power_curve",
        "direction_lookup",
        "cyclic_gam",
        "identity_linear",
        "kernel_linear",
        "pywake",
        "pywake_discrepancy",
        "windfold",
        "windfold_no_physics",
        "windfold_no_kernel",
        "windfold_no_local",
        "windfold_permuted_layout",
        "windfold_default_physics",
    }
    group = engine.groups(test["timestamp"].to_physical().to_numpy())
    scores = {}
    for spec in specs:
        fitted = (
            windfold
            if spec.name == "windfold"
            else registry.model_from_blocks(spec, SETTINGS).fit(train, context(farm))
        )
        out = fitted.predict(test)
        assert out.height == test.height
        mean = out["mean"].to_numpy()
        np.testing.assert_allclose(np.bincount(group, mean), 0.0, atol=1e-9)
        if spec.name == "direction_lookup":
            assert out.columns == ["mean", "variance", "support"]
            assert out["variance"].null_count() == test.height
        else:
            assert out.columns == ["mean", "variance"]
            assert isinstance(fitted, registry.FittedEngine)
            np.testing.assert_array_equal(fitted.predict_mean(test), mean)
            assert (out["variance"].to_numpy() > 0).all()
        scores[spec.name] = float(np.mean((test["y"].to_numpy() - mean) ** 2))
    assert scores["windfold"] < scores["no_wake"] and scores["kernel_linear"] < scores["no_wake"]


def test_stored_kernel_rows_equal_independent_basis_sums(farm):
    rows = pl.read_parquet(farm / "rows.parquet")
    stored = blocks.read_kernel_design(farm / "kernel_design")
    along_knots = BASIS.knot_vector(BASIS.along_knots)
    across_knots = BASIS.knot_vector(BASIS.across_knots)
    for r in (3, 40, 157, 600):
        row = rows.row(r, named=True)
        along, across = geometry.relations(EAST, NORTH, row["farm_direction"], DIAMETER)
        i = row["turbine_index"]
        expected = np.zeros(BASIS.columns)
        for j in range(len(EAST)):
            a, c = along[i, j], across[i, j]
            if not (
                (row["operating_mask"] >> j) & 1
                and geometry.SIDE_BY_SIDE_D < a <= 30
                and abs(c) <= 3
            ):
                continue
            for p in range(BASIS.along_size):
                bp = BSpline(along_knots, np.eye(BASIS.along_size)[p], 3)(a)
                for q in range(BASIS.across_size):
                    bq = BSpline(across_knots, np.eye(BASIS.across_size)[q], 3)(c)
                    expected[row["speed_bin"] * BASIS.per_bin + p * BASIS.across_size + q] += (
                        bp * bq
                    )
        np.testing.assert_allclose(stored[[r]].toarray()[0], expected, atol=1e-12)
    narrower = blocks.kernel_basis(20.0, 3.0, 1.5, 9, 0.45, 4, 3, 3)
    assert narrower.columns == BASIS.columns
    blocks.read_kernel_design(farm / "kernel_design", basis_key=BASIS.key())
    with pytest.raises(ValueError, match="another kernel basis or window"):
        blocks.read_kernel_design(farm / "kernel_design", basis_key=narrower.key())
    permuted = blocks.read_kernel_design(farm / "kernel_design" / "permuted")
    assert permuted.shape == stored.shape
    assert not np.array_equal(permuted.toarray(), stored.toarray())


def test_direction_lookup_is_the_window_mean_with_a_speed_bin_fallback(farm):
    train, _ = split(farm)
    spec = load_models()["direction_lookup"]
    fitted = registry.model_from_blocks(spec, SETTINGS).fit(train, context(farm))
    probe = train.filter(pl.col("turbine_index") == 1, pl.col("speed_bin") == 2)
    degree = np.mod(np.rint(probe["farm_direction"].to_numpy()), 360)
    for target in (250.0, 80.0):
        near = np.abs(((degree - target + 180) % 360) - 180) <= 5
        expected = probe["y"].to_numpy()[near].mean() if near.any() else probe["y"].mean()
        assert isinstance(fitted, registry.FittedLookup)
        got = fitted.table[1, 2, int(target)]
        if np.isnan(got):
            got = fitted.fallback[1, 2]
        assert got == pytest.approx(expected)
        query = probe.head(1).with_columns(farm_direction=pl.lit(target + 0.3))
        assert fitted.support(query)[0] == near.sum()
    predicted = fitted.predict(train)
    assert predicted.schema["support"] == pl.Int64 and predicted["support"].null_count() == 0
    np.testing.assert_array_equal(predicted["support"].to_numpy(), fitted.support(train))


def test_predictions_ignore_the_targets_own_speed_power_and_y(farm, windfold):
    _, test = split(farm)
    before = windfold.predict(test)
    changed = test.with_columns(
        speed=pl.col("speed") * 1.5, power_kw=pl.col("power_kw") * 0.5, y=pl.col("y") + 1.0
    )
    after = windfold.predict(changed)
    assert before.equals(after)


def test_pywake_picks_the_family_with_the_lowest_physics_only_mse(farm):
    train, _ = split(farm)
    fitted = registry.model_from_blocks(load_models()["pywake"], SETTINGS).fit(train, context(farm))
    assert isinstance(fitted, registry.FittedEngine)
    physics = registry.aligned(train, pl.read_parquet(farm / "physics.parquet"), "physics")
    y = train["y"].to_numpy()
    group = engine.groups(train["timestamp"].to_physical().to_numpy())
    offsets = np.zeros(len(EAST))
    scores = {}
    for family in ("jensen", "niayifar", "turbopark"):
        residual = y - engine.centre(physics[f"physics_{family}"].to_numpy(), group)
        design = np.eye(len(EAST))[train["turbine_index"].to_numpy()]
        centred = design - np.vstack([design[group == g].mean(0) for g in group])
        offsets = np.linalg.lstsq(centred, residual, rcond=None)[0]
        scores[family] = np.mean((residual - centred @ offsets) ** 2)
    assert fitted.layout.family == min(scores, key=lambda f: scores[f])


def test_variance_of_a_day_sum_adds_noise_and_unseen_cells_with_cross_covariances(farm):
    train, test = split(farm)
    stamps = test["timestamp"].unique().sort()
    rows = test.filter(pl.col("timestamp").is_in(stamps[:40].implode()))
    bucket = rows["direction_bucket"][0]
    hidden = pl.col("direction_bucket") == bucket
    spec = load_models()["windfold_no_kernel"]
    kept = train.filter(~hidden)
    fitted = registry.model_from_blocks(spec, SETTINGS).fit(kept, context(farm))
    assert isinstance(fitted, registry.FittedEngine)
    assert rows.filter(hidden).height > 0 and rows.filter(~hidden).height > 0
    trained = set(kept.select("direction_bucket", "speed_bin").unique().iter_rows())
    n = rows.height
    group = engine.groups(rows["timestamp"].to_physical().to_numpy())
    rng = np.random.default_rng(2)
    one_turbine = (rows["turbine_index"].to_numpy() == 0).astype(np.float64)
    weights = np.vstack([np.eye(n)[:5], np.ones(n), one_turbine, rng.normal(size=(2, n))])
    got = fitted.variance(rows, sp.csr_array(weights))
    centring = np.eye(n) - (group[:, None] == group[None, :]) / np.bincount(group)[group][:, None]
    noise = fitted.noise_variances[rows["speed_bin"].to_numpy().astype(np.int64)]
    keys = {}
    unseen = np.zeros((n, n))
    for k, row in enumerate(rows.iter_rows(named=True)):
        if (row["direction_bucket"], row["speed_bin"]) not in trained:
            key = (row["direction_bucket"], row["speed_bin"], row["turbine_index"])
            unseen[k, keys.setdefault(key, len(keys))] = 1.0
    tau = fitted.scales["local"]
    centred, local, _, _ = fitted.functionals(rows)
    posterior = engine.functional_variance(
        fitted.posterior,
        fitted.stats,
        sp.csr_array(sp.csr_array(weights) @ centred),
        sp.csr_array(sp.csr_array(weights) @ local) if local is not None else None,
    )
    joint = centring @ (np.diag(noise) + tau * unseen @ unseen.T) @ centring.T
    expected = posterior + np.einsum("ri,ij,rj->r", weights, joint, weights)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    assert abs(got[5]) < 1e-12 and got[6] > got[:5].max()
    np.testing.assert_allclose(fitted.predict(rows)["variance"].to_numpy()[:5], got[:5], rtol=1e-12)


def test_binned_kernel_is_one_indicator_per_knot_cell():
    binned = blocks.kernel_basis(30.0, 3.0, 1.5, 9, 0.45, 4, 0, 3)
    rng = np.random.default_rng(4)
    along = np.r_[rng.uniform(0.0, 30.0, 200), 0.0, 30.0, 1.5]
    across = np.r_[rng.uniform(-3.0, 3.0, 200), -3.0, 3.0, 0.0]
    values = binned.values(along, across).toarray()
    p = np.minimum(np.searchsorted(binned.along_knots, along, side="right") - 1, 8)
    q = np.minimum(np.searchsorted(binned.across_knots, across, side="right") - 1, 7)
    expected = np.zeros_like(values)
    expected[np.arange(len(along)), p * binned.across_size + q] = 1.0
    np.testing.assert_array_equal(values, expected)
    assert binned.per_bin == 9 * 8
    assert np.linalg.eigvalsh(binned.penalty(0.1)).min() > 0


@pytest.mark.parametrize("degree", [3, 0])
def test_a_row_without_upstream_neighbours_gets_a_zero_kernel_row(degree):
    basis = blocks.kernel_basis(30.0, 3.0, 1.5, 9, 0.45, 4, degree, 3)
    rows = pl.DataFrame(
        {
            "turbine_index": [0, 1],
            "operating_mask": [3, 3],
            "farm_direction": [0.0, 0.0],
            "speed_bin": [0, 0],
        }
    )
    design = blocks.kernel_design(
        rows, np.array([0.0, 300.0]), np.array([0.0, 0.0]), DIAMETER, basis, 30.0, 3.0
    )
    assert design.shape == (2, basis.columns) and design.nnz == 0


def test_learner_models_are_built_by_their_fit_jobs_not_the_registry():
    spec = load_models()["identity_gbm"]
    with pytest.raises(ValueError, match="its fit job builds it"):
        registry.build_model(spec, None, None, None)  # type: ignore[arg-type]


def test_engine_facts_carry_the_learned_kernel_and_local_cells(windfold):
    facts = json.loads(json.dumps(windfold.facts()))
    assert len(facts["blocks"]["shared_kernel"]) == BASIS.columns
    assert np.shape(facts["local_means"]) == (len(facts["local_cells"]), len(EAST))
    assert set(facts["scales"]) >= {"shared_kernel", "local"} and len(facts["noise"]) == 3


def test_a_tiny_physics_variance_holds_the_physics_scale_at_one(farm, windfold):
    train, _ = split(farm)
    settings = dataclasses.replace(SETTINGS, physics_variance=1e-8)
    held = registry.model_from_blocks(load_models()["windfold"], settings).fit(train, context(farm))
    assert isinstance(held, registry.FittedEngine) and SETTINGS.physics_variance == 0.25
    assert abs(windfold.facts()["blocks"]["physics"][0] - 1.0) > 0.05
    assert abs(held.facts()["blocks"]["physics"][0] - 1.0) < 1e-4


def test_measured_inflow_takes_phi_from_the_inflow_table(farm):
    train, _ = split(farm)
    specs = load_models()
    measured = registry.aligned(train, pl.read_parquet(farm / inflow.TABLE), "inflow")
    constant = registry.aligned(train, pl.read_parquet(farm / "physics.parquet"), "physics")
    diurnal = registry.aligned(train, pl.read_parquet(farm / inflow.TABLES["diurnal"]), "diurnal")
    for name, source in (
        ("pywake_inflow", measured),
        ("pywake", constant),
        ("windfold_default_physics", measured),
        ("windfold_diurnal_inflow", diurnal),
    ):
        spec = specs[name]
        phi = registry.read_inputs(train, farm, farm / "kernel_design", spec, BASIS).phi
        suffix = "_default" if spec.physics_parameters == "default" else ""
        for family, values in phi.items():
            np.testing.assert_array_equal(values, source[f"physics_{family}{suffix}"].to_numpy())
    for other in (constant, diurnal):
        assert not np.array_equal(
            measured["physics_niayifar"].to_numpy(), other["physics_niayifar"].to_numpy()
        )


def test_a_spec_scale_overrides_the_protocol_scale():
    protocol = load_protocol()
    specs = load_models()
    fixed = protocol.engine.fixed_prior_variance
    assert protocol.engine.scales == "evidence"
    assert (
        registry.engine_settings(protocol, specs["windfold"].scales).fixed_prior_variance == fixed
    )
    for name in ("windfold_evidence_scales", "kernel_linear"):
        settings = registry.engine_settings(protocol, specs[name].scales)
        assert settings.fixed_prior_variance is None
