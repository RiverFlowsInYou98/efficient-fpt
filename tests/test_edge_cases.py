import numpy as np

from efficient_fpt.models import SingleStageModel, aDDModel
from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.single_stage_cy import fptd_single_cy, q_single_cy
from efficient_fpt.multi_stage_cy import compute_loss_parallel, compute_loss_serial


def test_single_stage_simulate_fpt_respects_exact_T():
    model = SingleStageModel(mu=0.0, sigma=0.0, a=1.0, b=1.0, x0=0.0)
    rt, choice, x_final = model.simulate_fpt(1, T=0.95, dt=0.3, rng=0)

    assert rt[0] == -1.0
    assert choice[0] == 0
    assert x_final[0] == 0.0


def test_addm_simulate_fpt_respects_exact_T():
    model = aDDModel(eta=0.5, kappa=0.0, sigma=0.0, a=1.0, b=1.0, x0=0.0)
    rt, choice, x_final = model.simulate_fpt(
        np.array([1]),
        np.array([1]),
        np.array([0], dtype=np.int32),
        np.array([[0.0]], dtype=np.float64),
        np.array([1], dtype=np.int32),
        T=0.95,
        dt=0.3,
        rng=0,
    )

    assert rt[0] == -1.0
    assert choice[0] == 0
    assert x_final[0] == 0.0


def test_random_x0_uniform_is_reproducible():
    model = SingleStageModel(
        mu=0.0, sigma=1.0, a=1.0, b=0.0, x0={"dist_name": "uniform"}
    )
    rt1, ch1, xf1 = model.simulate_fpt(20, T=1.0, dt=0.1, rng=42)
    rt2, ch2, xf2 = model.simulate_fpt(20, T=1.0, dt=0.1, rng=42)

    np.testing.assert_array_equal(rt1, rt2)
    np.testing.assert_array_equal(ch1, ch2)
    np.testing.assert_array_equal(xf1, xf2)


def test_random_x0_beta_is_reproducible():
    model = SingleStageModel(
        mu=0.0,
        sigma=1.0,
        a=1.0,
        b=0.0,
        x0={"dist_name": "beta", "alpha": 2.0, "beta": 5.0},
    )
    rt1, ch1, xf1 = model.simulate_fpt(20, T=1.0, dt=0.1, rng=123)
    rt2, ch2, xf2 = model.simulate_fpt(20, T=1.0, dt=0.1, rng=123)

    np.testing.assert_array_equal(rt1, rt2)
    np.testing.assert_array_equal(ch1, ch2)
    np.testing.assert_array_equal(xf1, xf2)


def test_single_stage_out_of_domain_returns_zero():
    params = dict(mu=1.0, sigma=1.0, a1=1.5, b1=-0.3, a2=-1.5, b2=0.3, x0=0.0)

    assert fptd_single(0.0, **params, bdy=1) == 0.0
    assert fptd_single(10.0, **params, bdy=1) == 0.0
    assert fptd_single_cy(0.0, **params, bdy=1) == 0.0
    assert fptd_single_cy(10.0, **params, bdy=1) == 0.0

    assert q_single(2.0, **params, T=1.0) == 0.0
    assert q_single_cy(2.0, **params, T=1.0) == 0.0


def test_compute_loss_parallel_matches_serial_with_mixed_validity():
    mu1_data = np.array([0.2, -0.1, 0.0, 0.3, 0.0, 0.0], dtype=np.float64)
    mu2_data = np.zeros_like(mu1_data)
    rt_data = np.array([0.25, 0.4, 0.8, 1.2, 0.0, 10.0], dtype=np.float64)
    choice_data = np.array([1, -1, 1, -1, 1, 1], dtype=np.int32)
    sacc_array_data = np.zeros((len(rt_data), 1), dtype=np.float64)
    d_data = np.ones(len(rt_data), dtype=np.int32)

    serial = compute_loss_serial(
        mu1_data,
        mu2_data,
        rt_data,
        choice_data,
        sacc_array_data,
        d_data,
        1,
        1.0,
        1.5,
        0.3,
        0.0,
    )

    parallel_vals = [
        compute_loss_parallel(
            mu1_data,
            mu2_data,
            rt_data,
            choice_data,
            sacc_array_data,
            d_data,
            1,
            1.0,
            1.5,
            0.3,
            0.0,
            num_threads=2,
        )
        for _ in range(5)
    ]

    assert np.isfinite(serial)
    assert all(np.isclose(val, serial) for val in parallel_vals)


def test_compute_loss_returns_nan_when_all_trials_invalid():
    n = 8
    mu1_data = np.zeros(n, dtype=np.float64)
    mu2_data = np.zeros(n, dtype=np.float64)
    rt_data = np.zeros(n, dtype=np.float64)
    choice_data = np.ones(n, dtype=np.int32)
    sacc_array_data = np.zeros((n, 1), dtype=np.float64)
    d_data = np.ones(n, dtype=np.int32)

    serial = compute_loss_serial(
        mu1_data,
        mu2_data,
        rt_data,
        choice_data,
        sacc_array_data,
        d_data,
        1,
        1.0,
        1.5,
        0.3,
        0.0,
    )
    parallel = compute_loss_parallel(
        mu1_data,
        mu2_data,
        rt_data,
        choice_data,
        sacc_array_data,
        d_data,
        1,
        1.0,
        1.5,
        0.3,
        0.0,
        num_threads=2,
    )

    assert np.isnan(serial)
    assert np.isnan(parallel)
