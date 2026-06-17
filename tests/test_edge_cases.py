import warnings

import numpy as np
import pytest

pytest.importorskip("efpt.cython.single_stage")

from efpt.models import SingleStageModel, aDDModel
from efpt.single_stage import fptd_single, q_single
from efpt.cython.single_stage import (
    fptd_single as fptd_single_cy,
    q_single as q_single_cy,
)
from efpt.cython.batch import (
    compute_addm_loglikelihoods,
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
            invalid_policy="warn",
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
                invalid_policy="warn",
            )
            for _ in range(3)
        ]

    assert np.isfinite(serial)
    assert all(np.isclose(val, serial) for val in parallel_vals)


def test_compute_loss_returns_infinity_when_all_trials_are_negative_infinity():
    n = 8
    mu1_data = np.zeros(n, dtype=np.float64)
    mu2_data = np.zeros(n, dtype=np.float64)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.full(n, 100.0, dtype=np.float64)
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
            invalid_policy="inf",
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
            invalid_policy="inf",
        )

    assert np.isinf(serial)
    assert np.isinf(parallel)


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
        "trial 0 outputs -inf log-likelihood",
        "trial 2 outputs -inf log-likelihood",
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
            invalid_policy="warn",
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
            invalid_policy="warn",
        )

    assert np.isfinite(serial)
    assert parallel == pytest.approx(serial)
    assert [str(w.message) for w in serial_warnings] == expected
    assert [str(w.message) for w in parallel_warnings] == expected


def test_jax_nll_matches_cython_warn_policy_and_warns_in_order():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from efpt.jax.batch import compute_addm_nll

    mu1_data = np.array([0.2, 0.0, -0.1, 0.0], dtype=np.float64)
    mu2_data = np.zeros_like(mu1_data)
    eta, kappa, r1_data, r2_data, flag_data = _public_addm_covariates(
        mu1_data, mu2_data
    )
    rt_data = np.array([0.25, 10.0, 0.4, 10.0], dtype=np.float64)
    choice_data = np.array([1, 1, -1, -1], dtype=np.int32)
    sacc_array_data = np.zeros((len(rt_data), 1), dtype=np.float64)
    d_data = np.ones(len(rt_data), dtype=np.int32)

    loglikelihoods = compute_addm_loglikelihoods(
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
    valid_count = int(np.sum(np.isfinite(loglikelihoods)))

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
            invalid_policy="warn",
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
            invalid_policy="warn",
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
                invalid_policy="warn",
                warn=True,
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
                invalid_policy="warn",
                warn=True,
            )
        )

    assert mean_nll == pytest.approx(serial)
    assert mean_nll == pytest.approx(parallel)
    assert sum_nll == pytest.approx(serial * valid_count)
    assert [str(w.message) for w in caught] == [
        "trial 1 outputs -inf log-likelihood",
        "trial 3 outputs -inf log-likelihood",
        "trial 1 outputs -inf log-likelihood",
        "trial 3 outputs -inf log-likelihood",
    ]


def test_jax_nll_returns_infinity_when_all_loglikelihoods_are_negative_infinity():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from efpt.jax.batch import compute_addm_nll

    rt_data = jnp.full(3, 100.0, dtype=jnp.float64)
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
                invalid_policy="inf",
                warn=True,
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
                invalid_policy="inf",
                warn=True,
            )
        )

    assert np.isinf(mean_nll)
    assert np.isinf(sum_nll)
    assert [str(w.message) for w in caught] == [
        "trial 0 outputs -inf log-likelihood",
        "trial 1 outputs -inf log-likelihood",
        "trial 2 outputs -inf log-likelihood",
        "trial 0 outputs -inf log-likelihood",
        "trial 1 outputs -inf log-likelihood",
        "trial 2 outputs -inf log-likelihood",
    ]


def test_jax_one_shot_nll_warns_by_default_but_closure_stays_silent():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from efpt.jax.batch import compute_addm_nll, make_addm_nll_function

    rt_data = np.array([10.0, 0.4, 10.0], dtype=np.float64)
    choice_data = np.ones(3, dtype=np.int32)
    r1_data = np.zeros(3, dtype=np.float64)
    r2_data = np.zeros(3, dtype=np.float64)
    flag_data = np.zeros(3, dtype=np.int32)
    sacc_array_data = np.zeros((3, 1), dtype=np.float64)
    d_data = np.ones(3, dtype=np.int32)

    with warnings.catch_warnings(record=True) as one_shot_warnings:
        warnings.simplefilter("always")
        loss = float(
            compute_addm_nll(
                jnp.array(rt_data),
                jnp.array(choice_data),
                0.0,
                1.0,
                1.0,
                1.5,
                0.3,
                0.0,
                jnp.array(r1_data),
                jnp.array(r2_data),
                jnp.array(flag_data),
                jnp.array(sacc_array_data),
                jnp.array(d_data),
                invalid_policy="warn",
            )
        )

    nll_fn = make_addm_nll_function(
        jnp.array(rt_data),
        jnp.array(choice_data),
        jnp.array(r1_data),
        jnp.array(r2_data),
        jnp.array(flag_data),
        jnp.array(sacc_array_data),
        jnp.array(d_data),
        invalid_policy="warn",
    )

    with warnings.catch_warnings(record=True) as closure_warnings:
        warnings.simplefilter("always")
        closure_loss = float(nll_fn(0.0, 1.0, 1.0, 1.5, 0.3, 0.0))

    assert np.isfinite(loss)
    assert closure_loss == pytest.approx(loss)
    assert [str(w.message) for w in one_shot_warnings] == [
        "trial 0 outputs -inf log-likelihood",
        "trial 2 outputs -inf log-likelihood",
    ]
    assert closure_warnings == []


def test_tada_warn_policy_skips_negative_infinity_trials():
    n = 4
    rt_data = np.array([0.4, 0.0, 0.6, 0.0], dtype=np.float64)
    choice_data = np.array([1, 1, -1, -1], dtype=np.int32)
    r1_data = np.zeros(n, dtype=np.float64)
    r2_data = np.zeros(n, dtype=np.float64)
    flag_data = np.zeros(n, dtype=np.int32)
    sacc_array_data = np.zeros((n, 1), dtype=np.float64)
    d_data = np.ones(n, dtype=np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = compute_tada_mean_nll(
            rt_data,
            choice_data,
            0.0,
            1.0,
            1.0,
            1.5,
            0.0,
            0.0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            invalid_policy="warn",
        )

    assert np.isfinite(loss)
