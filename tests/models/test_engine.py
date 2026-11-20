from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import block_diag, null_space
from scipy.stats import multivariate_normal

from windfold.models import engine
from windfold.models.blocks import second_difference

GAUSSIAN: Any = multivariate_normal

BOUNDS = engine.Bounds(noise_floor=1e-6, variance_floor=1e-8, variance_ceiling=100.0)
TURBINES, COLUMNS, BINS, BUCKETS = 4, 6, 2, 3


def toy(seed: int, timestamps: int = 40):
    rng = np.random.default_rng(seed)
    group, turbine, speed_bin, cell = [], [], [], []
    for t in range(timestamps):
        members = np.sort(rng.choice(TURBINES, rng.integers(2, TURBINES + 1), replace=False))
        k = int(rng.integers(0, BINS))
        bucket = int(rng.integers(0, BUCKETS))
        group += [t] * len(members)
        turbine += members.tolist()
        speed_bin += [k] * len(members)
        cell += [bucket * BINS + k] * len(members)
    group, turbine, speed_bin, cell = (
        np.array(a, dtype=np.int64) for a in (group, turbine, speed_bin, cell)
    )
    design = rng.normal(size=(len(group), COLUMNS))
    design[:, :2] = np.eye(2)[turbine % 2]
    raw = design @ np.array([0.1, -0.1, 0.8, 0.3, -0.2, 0.1]) + rng.normal(0, 0.2, len(group))
    y = engine.centre(raw, group)
    return group, turbine, speed_bin, cell, design, y


def priors() -> list[engine.Prior]:
    return [
        engine.Prior("offset", slice(0, 2), np.eye(2), np.zeros(2), 1.0),
        engine.Prior("physics", slice(2, 3), np.eye(1), np.ones(1), 0.25),
        engine.Prior(
            "kernel", slice(3, 6), second_difference(3) + 0.1 * np.eye(3), np.zeros(3), None
        ),
    ]


def local_columns(cell, turbine, cells):
    index = np.searchsorted(cells, cell) * TURBINES + turbine
    return np.eye(len(cells) * TURBINES)[index]


def brute_force(theta, group, turbine, speed_bin, cell, design, y):
    noise = np.exp(theta[:BINS])
    kernel_variance, local_variance = np.exp(theta[BINS]), np.exp(theta[BINS + 1])
    cells = np.unique(cell)
    full = np.hstack([design, local_columns(cell, turbine, cells)])
    z, rows, variance = [], [], []
    for t in np.unique(group):
        members = group == t
        q = null_space(np.ones((1, members.sum())))
        z.append(q.T @ y[members])
        rows.append(q.T @ full[members])
        variance += [noise[speed_bin[members][0]]] * q.shape[1]
    z, z_design = np.concatenate(z), np.vstack(rows)
    precision = np.asarray(
        block_diag(
            np.eye(2),
            np.eye(1) / 0.25,
            (second_difference(3) + 0.1 * np.eye(3)) / kernel_variance,
            np.eye(len(cells) * TURBINES) / local_variance,
        ),
        dtype=np.float64,
    )
    mean = np.r_[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.zeros(len(cells) * TURBINES)]
    covariance = z_design @ np.linalg.inv(precision) @ z_design.T + np.diag(variance)
    log_evidence = GAUSSIAN(z_design @ mean, covariance).logpdf(z)
    inverse_noise = np.diag(1.0 / np.array(variance))
    posterior_precision = z_design.T @ inverse_noise @ z_design + precision
    posterior_mean = np.linalg.solve(
        posterior_precision, z_design.T @ inverse_noise @ z + precision @ mean
    )
    return log_evidence, posterior_mean, posterior_precision, full


def fitted_engine(group, turbine, speed_bin, cell, design, y):
    stats = engine.statistics(
        sp.csr_array(design), y, group, speed_bin, BINS, turbine, cell, TURBINES
    )
    return engine.Engine(stats, priors(), True, BOUNDS), stats


THETA = np.log([0.03, 0.05, 0.4, 0.02])


def test_projection_equals_the_contrast_likelihood_and_the_dense_posterior():
    data = toy(seed=1)
    model, stats = fitted_engine(*data)
    log_evidence, mean, _, _ = brute_force(THETA, *data)
    assert -model.evaluate(THETA, gradient=False)[0] == pytest.approx(log_evidence, rel=1e-10)
    posterior = model.posterior(THETA)
    np.testing.assert_allclose(posterior.mean, mean[:COLUMNS], rtol=1e-9, atol=1e-12)
    unrotated = np.einsum("ctk,ck->ct", stats.vectors, posterior.local.reshape(-1, TURBINES))
    np.testing.assert_allclose(unrotated.ravel(), mean[COLUMNS:], rtol=1e-8, atol=1e-10)


def test_schur_path_equals_the_dense_path():
    group, turbine, speed_bin, cell, design, y = toy(seed=2)
    cells = np.unique(cell)
    dense_design = np.hstack([design, local_columns(cell, turbine, cells)])
    local = len(cells) * TURBINES
    dense_priors = priors() + [
        engine.Prior("local", slice(COLUMNS, COLUMNS + local), np.eye(local), np.zeros(local), None)
    ]
    stats = engine.statistics(sp.csr_array(dense_design), y, group, speed_bin, BINS)
    dense = engine.Engine(stats, dense_priors, False, BOUNDS)
    schur, _ = fitted_engine(group, turbine, speed_bin, cell, design, y)
    for theta in (THETA, np.log([0.1, 0.01, 2.0, 0.5])):
        a, grad_a = schur.evaluate(theta)
        b, grad_b = dense.evaluate(theta)
        assert a == pytest.approx(b, rel=1e-10)
        np.testing.assert_allclose(grad_a, grad_b, rtol=1e-7, atol=1e-9)


def test_gradient_equals_finite_differences():
    model, _ = fitted_engine(*toy(seed=3))
    _, grad = model.evaluate(THETA)
    step = 1e-6
    numeric = [
        (model.evaluate(THETA + step * e, False)[0] - model.evaluate(THETA - step * e, False)[0])
        / (2 * step)
        for e in np.eye(len(THETA))
    ]
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)


def test_fit_reaches_a_stationary_point_of_the_evidence():
    model, _ = fitted_engine(*toy(seed=4, timestamps=200))
    posterior = model.fit((1e-3, 1e-1, 10.0))
    theta = np.log(np.r_[posterior.noise, posterior.variances["kernel"], posterior.local_variance])
    _, grad = model.evaluate(theta)
    low, high = np.array(model.scale_bounds()).T
    inside = (theta > low + 1e-6) & (theta < high - 1e-6)
    assert np.all(np.abs(grad[inside]) < 1e-3)
    assert posterior.log_evidence == pytest.approx(-model.evaluate(theta, False)[0])


def test_variance_of_a_day_sum_equals_the_dense_answer():
    data = toy(seed=5)
    group, turbine, speed_bin, cell, design, y = data
    model, stats = fitted_engine(*data)
    posterior = model.posterior(THETA)
    _, _, precision, full = brute_force(THETA, *data)
    centred = full - np.vstack([full[group == t].mean(0) for t in group])
    day = np.isin(group, [3, 4, 5, 9])
    functional = centred[day].sum(0, keepdims=True)
    single = centred[[7, 11]]
    for f in (functional, single):
        expected = np.einsum("ri,ri->r", f, np.linalg.solve(precision, f.T).T)
        got = engine.functional_variance(
            posterior, stats, sp.csr_array(f[:, :COLUMNS]), sp.csr_array(f[:, COLUMNS:])
        )
        np.testing.assert_allclose(got, expected, rtol=1e-9)
