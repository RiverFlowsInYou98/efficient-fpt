import warnings
from pathlib import Path
import numpy as np
import pytest

from efficient_fpt.models import SingleStageModel, aDDModel
from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.cython.single_stage import fptd_single as fptd_single_cy, q_single as q_single_cy
from efficient_fpt.cython.batch import (
    compute_addm_likelihoods,
    compute_addm_mean_nll,
    compute_tada_mean_nll,
)


from helpers import mu_to_addm_covariates as _public_addm_covariates


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


def test_compute_addm_mean_nll_matches_serial_with_mixed_validity():
    mu1_data = np.array([0.2, -0.1, 0.0, 0.3, 0.0, 0.0], dtype=np.float64)
    mu2_data = np.zeros_like(mu1_data)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.array([0.25, 0.4, 0.8, 1.2, 0.0, 10.0], dtype=np.float64)
    choice_data = np.array([1, -1, 1, -1, 1, 1], dtype=np.int32)
    sacc_array_data = np.zeros((len(rt_data), 1), dtype=np.float64)
    d_data = np.ones(len(rt_data), dtype=np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
        )
        parallel_vals = [
            compute_addm_mean_nll(
                rt_data,
                choice_data,
                eta,
                kappa,
                1.0,
                1.5,
                0.3,
                0.0,
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                n_threads=2,
            )
            for _ in range(5)
        ]

    assert np.isfinite(serial)
    assert all(np.isclose(val, serial) for val in parallel_vals)


def test_compute_loss_returns_nan_when_all_trials_invalid():
    n = 8
    mu1_data = np.zeros(n, dtype=np.float64)
    mu2_data = np.zeros(n, dtype=np.float64)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.zeros(n, dtype=np.float64)
    choice_data = np.ones(n, dtype=np.int32)
    sacc_array_data = np.zeros((n, 1), dtype=np.float64)
    d_data = np.ones(n, dtype=np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
        )
        parallel = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=2,
        )

    assert np.isnan(serial)
    assert np.isnan(parallel)


def test_cython_loss_warnings_use_true_trial_indices():
    mu1_data = np.array([0.0, 0.2, 0.0], dtype=np.float64)
    mu2_data = np.zeros_like(mu1_data)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.array([10.0, 0.4, 10.0], dtype=np.float64)
    choice_data = np.ones(len(rt_data), dtype=np.int32)
    sacc_array_data = np.zeros((len(rt_data), 1), dtype=np.float64)
    d_data = np.ones(len(rt_data), dtype=np.int32)
    expected = [
        "trial 0 outputs 0 likelihood, skipped",
        "trial 2 outputs 0 likelihood, skipped",
    ]

    with warnings.catch_warnings(record=True) as serial_warnings:
        warnings.simplefilter("always")
        serial = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
        )

    with warnings.catch_warnings(record=True) as parallel_warnings:
        warnings.simplefilter("always")
        parallel = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=2,
        )

    assert np.isfinite(serial)
    assert parallel == pytest.approx(serial)
    assert [str(w.message) for w in serial_warnings] == expected
    assert [str(w.message) for w in parallel_warnings] == expected


def test_jax_nll_matches_cython_skip_only_and_warns_in_order():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from efficient_fpt.jax.batch import compute_addm_nll

    mu1_data = np.array([0.2, 0.0, -0.1, 0.0], dtype=np.float64)
    mu2_data = np.zeros_like(mu1_data)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.array([0.25, 10.0, 0.4, 10.0], dtype=np.float64)
    choice_data = np.array([1, 1, -1, -1], dtype=np.int32)
    sacc_array_data = np.zeros((len(rt_data), 1), dtype=np.float64)
    d_data = np.ones(len(rt_data), dtype=np.int32)

    likelihoods = compute_addm_likelihoods(
        rt_data,
        choice_data,
        eta,
        kappa,
        1.0,
        1.5,
        0.3,
        0.0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
    )
    valid_count = int(np.sum(likelihoods > 0.0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
        )
        parallel = compute_addm_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=2,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean_nll = float(
            compute_addm_nll(
                jnp.array(rt_data),
                jnp.array(choice_data),
                eta,
                kappa,
                1.0,
                1.5,
                0.3,
                0.0,
                jnp.array(r1_data),
                jnp.array(r2_data),
                jnp.array(flag_data),
                jnp.array(sacc_array_data),
                jnp.array(d_data),
                reduce="mean",
            )
        )
        sum_nll = float(
            compute_addm_nll(
                jnp.array(rt_data),
                jnp.array(choice_data),
                eta,
                kappa,
                1.0,
                1.5,
                0.3,
                0.0,
                jnp.array(r1_data),
                jnp.array(r2_data),
                jnp.array(flag_data),
                jnp.array(sacc_array_data),
                jnp.array(d_data),
                reduce="sum",
            )
        )

    assert mean_nll == pytest.approx(serial)
    assert mean_nll == pytest.approx(parallel)
    assert sum_nll == pytest.approx(serial * valid_count)
    assert [str(w.message) for w in caught] == [
        "trial 1 outputs 0 likelihood, skipped",
        "trial 3 outputs 0 likelihood, skipped",
        "trial 1 outputs 0 likelihood, skipped",
        "trial 3 outputs 0 likelihood, skipped",
    ]


def test_jax_nll_returns_nan_when_all_likelihoods_invalid():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from efficient_fpt.jax.batch import compute_addm_nll

    rt_data = jnp.zeros(3, dtype=jnp.float64)
    choice_data = jnp.ones(3, dtype=jnp.int32)
    r1_data = jnp.zeros(3, dtype=jnp.float64)
    r2_data = jnp.zeros(3, dtype=jnp.float64)
    flag_data = jnp.zeros(3, dtype=jnp.int32)
    sacc_array_data = jnp.zeros((3, 1), dtype=jnp.float64)
    d_data = jnp.ones(3, dtype=jnp.int32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean_nll = float(
            compute_addm_nll(
                rt_data,
                choice_data,
                0.0,
                1.0,
                1.0,
                1.5,
                0.3,
                0.0,
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                reduce="mean",
            )
        )
        sum_nll = float(
            compute_addm_nll(
                rt_data,
                choice_data,
                0.0,
                1.0,
                1.0,
                1.5,
                0.3,
                0.0,
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                reduce="sum",
            )
        )

    assert np.isnan(mean_nll)
    assert np.isnan(sum_nll)
    assert [str(w.message) for w in caught] == [
        "trial 0 outputs 0 likelihood, skipped",
        "trial 1 outputs 0 likelihood, skipped",
        "trial 2 outputs 0 likelihood, skipped",
        "trial 0 outputs 0 likelihood, skipped",
        "trial 1 outputs 0 likelihood, skipped",
        "trial 2 outputs 0 likelihood, skipped",
    ]


def test_make_nll_function_skips_invalid_silently():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from efficient_fpt.jax.batch import make_addm_nll_function

    rt_data = jnp.array([0.4, 10.0, 0.6], dtype=jnp.float64)
    choice_data = jnp.array([1, 1, -1], dtype=jnp.int32)
    r1_data = jnp.array([0.2, 0.0, -0.2], dtype=jnp.float64)
    r2_data = jnp.zeros(3, dtype=jnp.float64)
    flag_data = jnp.zeros(3, dtype=jnp.int32)
    sacc_array_data = jnp.zeros((3, 1), dtype=jnp.float64)
    d_data = jnp.ones(3, dtype=jnp.int32)
    nll_fn = make_addm_nll_function(
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order=30,
        trunc_num=50,
        log_space=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss = float(nll_fn(0.0, 1.0, 1.0, 1.5, 0.3, 0.0))

    assert np.isfinite(loss)
    assert len(caught) == 0


def test_repo_no_longer_references_safe_sacc_helper():
    forbidden = ("pad_sacc_array_safely", "safe_sacc")
    targets = [
        Path("src/efficient_fpt/jax/__init__.py"),
        Path("src/efficient_fpt/jax/multi_stage.py"),
        Path("examples/tutorial_numpy_vs_jax.ipynb"),
        Path("examples/pymc_sampler_comparison.ipynb"),
        Path("examples/inference_comparison.ipynb"),
    ]

    for path in targets:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{token} still present in {path}"


def test_tada_parallel_respects_num_threads_and_warning_messages():
    mu1_data = np.array([0.2, 0.1, 0.0, -0.2], dtype=np.float64)
    mu2_data = np.zeros_like(mu1_data)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.array([0.4, 0.0, 10.0, 0.6], dtype=np.float64)
    choice_data = np.array([1, 1, 1, -1], dtype=np.int32)
    sacc_array_data = np.zeros((len(rt_data), 1), dtype=np.float64)
    d_data = np.ones(len(rt_data), dtype=np.int32)
    expected = [
        "trial 1 has nonpositive rt, skipped",
        "trial 2 outputs 0 likelihood, skipped",
    ]

    with warnings.catch_warnings(record=True) as one_thread_warnings:
        warnings.simplefilter("always")
        loss_one = compute_tada_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=1,
        )

    with warnings.catch_warnings(record=True) as two_thread_warnings:
        warnings.simplefilter("always")
        loss_two = compute_tada_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=2,
        )

    assert np.isfinite(loss_one)
    assert loss_two == pytest.approx(loss_one)
    assert [str(w.message) for w in one_thread_warnings] == expected
    assert [str(w.message) for w in two_thread_warnings] == expected


def test_tada_returns_nan_when_all_trials_invalid():
    n = 4
    mu1_data = np.zeros(n, dtype=np.float64)
    mu2_data = np.zeros(n, dtype=np.float64)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.zeros(n, dtype=np.float64)
    choice_data = np.ones(n, dtype=np.int32)
    sacc_array_data = np.zeros((n, 1), dtype=np.float64)
    d_data = np.ones(n, dtype=np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_one = compute_tada_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=1,
        )
        loss_two = compute_tada_mean_nll(
            rt_data,
            choice_data,
            eta,
            kappa,
            1.0,
            1.5,
            0.3,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            n_threads=2,
        )

    assert np.isnan(loss_one)
    assert np.isnan(loss_two)
