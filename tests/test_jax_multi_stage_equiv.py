"""Equivalence and regression tests for the public JAX APIs."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax import grad, value_and_grad

from efpt.jax.batch import (
    compute_addm_loglikelihoods,
    compute_addm_loglikelihoods_batchscan,
    compute_addm_loglikelihoods_batchvmap,
    compute_addm_loglikelihoods_jit,
    make_addm_nll_function,
    make_addm_nll_function_batchscan,
    make_addm_nll_function_batchvmap,
)
from efpt.jax.multi_stage import (
    compute_addm_logfptd,
    compute_addm_logfptd_stagescan,
    compute_addm_logfptd_jit,
    compute_heterog_multistage_logfptd,
    compute_heterog_multistage_logfptd_stagescan,
    compute_heterog_multistage_logfptd_jit,
)


def _addm_public_batch_inputs():
    rt_data = jnp.array([0.9, 1.5, 2.0], dtype=jnp.float64)
    choice_data = jnp.array([1, -1, 1], dtype=jnp.int32)
    r1_data = jnp.array([0.45, 0.30, 0.20], dtype=jnp.float64)
    r2_data = jnp.array([0.10, 0.55, 0.40], dtype=jnp.float64)
    flag_data = jnp.array([0, 1, 0], dtype=jnp.int32)
    sacc_array_data = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0],
            [0.0, 0.6, 1.3, 0.0],
        ],
        dtype=jnp.float64,
    )
    d_data = jnp.array([1, 2, 3], dtype=jnp.int32)
    params = dict(eta=0.25, kappa=1.1, sigma=1.0, a=1.6, b=0.25, x0=0.0)
    return (
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        params,
    )


def _addm_public_batch_inputs_max_d2():
    rt_data = jnp.array([0.9, 1.4, 1.8], dtype=jnp.float64)
    choice_data = jnp.array([1, -1, 1], dtype=jnp.int32)
    r1_data = jnp.array([0.45, 0.30, 0.20], dtype=jnp.float64)
    r2_data = jnp.array([0.10, 0.55, 0.40], dtype=jnp.float64)
    flag_data = jnp.array([0, 1, 0], dtype=jnp.int32)
    sacc_array_data = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.7],
            [0.0, 0.8],
        ],
        dtype=jnp.float64,
    )
    d_data = jnp.array([1, 2, 2], dtype=jnp.int32)
    params = dict(eta=0.25, kappa=1.1, sigma=1.0, a=1.6, b=0.25, x0=0.0)
    return (
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        params,
    )


class TestJaxAddmBatch:
    def test_compute_addm_loglikelihoods_matches_scalar_loop(self):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs()

        batch_vals = np.asarray(
            compute_addm_loglikelihoods(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order=20,
                trunc_num=25,
                log_space=False,
            )
        )

        scalar_vals = np.asarray(
            [
                compute_addm_logfptd(
                    float(rt_data[i]),
                    int(choice_data[i]),
                    params["eta"],
                    params["kappa"],
                    params["sigma"],
                    params["a"],
                    params["b"],
                    params["x0"],
                    float(r1_data[i]),
                    float(r2_data[i]),
                    int(flag_data[i]),
                    sacc_array_data[i],
                    int(d_data[i]),
                    order=20,
                    trunc_num=25,
                    log_space=False,
                )
                for i in range(len(rt_data))
            ]
        )

        np.testing.assert_allclose(batch_vals, scalar_vals, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize(
        "fn",
        [compute_addm_loglikelihoods_batchvmap, compute_addm_loglikelihoods_batchscan],
    )
    def test_use_remat_preserves_batch_loglikelihoods(self, fn):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs()

        base = np.asarray(
            fn(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order=20,
                trunc_num=25,
                log_space=True,
                use_remat=False,
            )
        )
        remat = np.asarray(
            fn(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order=20,
                trunc_num=25,
                log_space=True,
                use_remat=True,
            )
        )

        np.testing.assert_allclose(remat, base, rtol=1e-8, atol=1e-10)

    def test_compute_addm_loglikelihoods_jit_matches_nonjit(self):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs()

        expected = np.asarray(
            compute_addm_loglikelihoods(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order=20,
                trunc_num=25,
                log_space=True,
            )
        )
        actual = np.asarray(
            compute_addm_loglikelihoods_jit(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order=20,
                trunc_num=25,
                log_space=True,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("log_space", [False, True])
    def test_batchscan_max_d_two_matches_scalar_loop_and_jit(self, log_space):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs_max_d2()

        expected = np.asarray(
            [
                compute_addm_logfptd(
                    float(rt_data[i]),
                    int(choice_data[i]),
                    params["eta"],
                    params["kappa"],
                    params["sigma"],
                    params["a"],
                    params["b"],
                    params["x0"],
                    float(r1_data[i]),
                    float(r2_data[i]),
                    int(flag_data[i]),
                    sacc_array_data[i],
                    int(d_data[i]),
                    order_mid=14,
                    order_last=18,
                    trunc_num=25,
                    log_space=log_space,
                )
                for i in range(len(rt_data))
            ]
        )

        actual = np.asarray(
            compute_addm_loglikelihoods_batchscan(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order_mid=14,
                order_last=18,
                trunc_num=25,
                log_space=log_space,
            )
        )
        jit_actual = np.asarray(
            compute_addm_loglikelihoods_jit(
                rt_data,
                choice_data,
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order_mid=14,
                order_last=18,
                trunc_num=25,
                log_space=log_space,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(jit_actual, expected, rtol=1e-8, atol=1e-10)

    def test_make_addm_nll_function_matches_legacy_vmap_gradients(self):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs()

        legacy = make_addm_nll_function_batchvmap(
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order=20,
            trunc_num=25,
            log_space=True,
        )
        current = make_addm_nll_function(
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order=20,
            trunc_num=25,
            log_space=True,
        )

        param_vec = jnp.array(
            [
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
            ],
            dtype=jnp.float64,
        )

        legacy_loss, legacy_grad = value_and_grad(
            lambda p: legacy(p[0], p[1], p[2], p[3], p[4], p[5])
        )(param_vec)
        current_loss, current_grad = value_and_grad(
            lambda p: current(p[0], p[1], p[2], p[3], p[4], p[5])
        )(param_vec)

        np.testing.assert_allclose(
            np.asarray(current_loss), np.asarray(legacy_loss), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(current_grad), np.asarray(legacy_grad), rtol=1e-7, atol=1e-9
        )

    @pytest.mark.parametrize(
        "factory",
        [make_addm_nll_function_batchvmap, make_addm_nll_function_batchscan],
    )
    def test_use_remat_preserves_nll_factory_values_and_gradients(self, factory):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs()

        base_fn = factory(
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order=20,
            trunc_num=25,
            log_space=True,
            use_remat=False,
        )
        remat_fn = factory(
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order=20,
            trunc_num=25,
            log_space=True,
            use_remat=True,
        )

        param_vec = jnp.array(
            [
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
            ],
            dtype=jnp.float64,
        )

        base_loss, base_grad = value_and_grad(
            lambda p: base_fn(p[0], p[1], p[2], p[3], p[4], p[5])
        )(param_vec)
        remat_loss, remat_grad = value_and_grad(
            lambda p: remat_fn(p[0], p[1], p[2], p[3], p[4], p[5])
        )(param_vec)

        np.testing.assert_allclose(
            np.asarray(remat_loss), np.asarray(base_loss), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(remat_grad), np.asarray(base_grad), rtol=1e-7, atol=1e-9
        )

    def test_invalid_policy_inf_returns_infinity(self):
        (
            rt_data,
            choice_data,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            params,
        ) = _addm_public_batch_inputs()
        bad_rt_data = rt_data.at[1].set(50.0)

        loss = float(
            make_addm_nll_function(
                bad_rt_data,
                choice_data,
                r1_data,
                r2_data,
                flag_data,
                sacc_array_data,
                d_data,
                order=20,
                trunc_num=25,
                log_space=True,
                invalid_policy="inf",
            )(
                params["eta"],
                params["kappa"],
                params["sigma"],
                params["a"],
                params["b"],
                params["x0"],
            )
        )
        assert np.isinf(loss)


class TestJaxScalarJit:
    def test_compute_addm_logfptd_jit_matches_nonjit(self):
        sacc_array = jnp.array([0.0, 0.7, 1.2, 0.0], dtype=jnp.float64)
        expected = float(
            compute_addm_logfptd(
                1.7,
                1,
                0.25,
                1.1,
                1.0,
                1.5,
                0.3,
                0.0,
                0.4,
                0.2,
                0,
                sacc_array,
                3,
                order=30,
                trunc_num=25,
                log_space=True,
            )
        )
        actual = float(
            compute_addm_logfptd_jit(
                1.7,
                1,
                0.25,
                1.1,
                1.0,
                1.5,
                0.3,
                0.0,
                0.4,
                0.2,
                0,
                sacc_array,
                3,
                order=30,
                trunc_num=25,
                log_space=True,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("log_space", [False, True])
    def test_addm_d2_precomputed_matches_stagescan_with_split_orders(self, log_space):
        sacc_array = jnp.array([0.0, 0.7], dtype=jnp.float64)
        expected = float(
            compute_addm_logfptd(
                1.4,
                1,
                0.25,
                1.1,
                1.0,
                1.5,
                0.3,
                0.0,
                0.4,
                0.2,
                0,
                sacc_array,
                2,
                order_mid=14,
                order_last=18,
                trunc_num=25,
                log_space=log_space,
            )
        )
        actual = float(
            compute_addm_logfptd_stagescan(
                1.4,
                1,
                0.25,
                1.1,
                1.0,
                1.5,
                0.3,
                0.0,
                0.4,
                0.2,
                0,
                sacc_array,
                2,
                order_mid=14,
                order_last=18,
                trunc_num=25,
                log_space=log_space,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("log_space", [False, True])
    def test_generalized_d2_precomputed_matches_stagescan_with_split_orders(
        self, log_space
    ):
        mu_array = jnp.array([0.4, -0.2], dtype=jnp.float64)
        node_array = jnp.array([0.0, 0.6], dtype=jnp.float64)
        sigma_array = jnp.array([1.0, 0.9], dtype=jnp.float64)
        b1_array = jnp.array([-0.3, -0.15], dtype=jnp.float64)
        b2_array = jnp.array([0.2, 0.1], dtype=jnp.float64)

        expected = float(
            compute_heterog_multistage_logfptd(
                1.5,
                1,
                0.0,
                1.5,
                -1.4,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                2,
                order_mid=14,
                order_last=18,
                trunc_num=25,
                log_space=log_space,
            )
        )
        actual = float(
            compute_heterog_multistage_logfptd_stagescan(
                1.5,
                1,
                0.0,
                1.5,
                -1.4,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                2,
                order_mid=14,
                order_last=18,
                trunc_num=25,
                log_space=log_space,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


class TestJaxScalarGradients:
    def test_compute_heterog_multistage_logfptd_grad_matches_finite_difference(self):
        mu_array = jnp.array([0.35, -0.15, 0.08, 0.0, 0.0], dtype=jnp.float64)
        node_array = jnp.array([0.0, 0.55, 1.25, 0.0, 0.0], dtype=jnp.float64)
        sigma_array = jnp.array([1.0, 0.9, 1.1, 1.0, 1.0], dtype=jnp.float64)
        b1_array = jnp.array([-0.22, -0.12, -0.05, 0.0, 0.0], dtype=jnp.float64)
        b2_array = jnp.array([0.18, 0.14, 0.06, 0.0, 0.0], dtype=jnp.float64)
        x0 = 0.05
        a2 = -1.35
        eps = 1e-5

        def f(a1):
            return compute_heterog_multistage_logfptd(
                1.75,
                1,
                x0,
                a1,
                a2,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                3,
                order=24,
                trunc_num=30,
                log_space=True,
            )

        autodiff_grad = float(grad(f)(1.45))
        finite_diff_grad = float((f(1.45 + eps) - f(1.45 - eps)) / (2.0 * eps))

        np.testing.assert_allclose(
            autodiff_grad, finite_diff_grad, rtol=5e-4, atol=1e-7
        )

    def test_compute_heterog_multistage_logfptd_jit_matches_nonjit(self):
        mu_array = jnp.array([0.4, -0.2, 0.1, 0.0, 0.0], dtype=jnp.float64)
        node_array = jnp.array([0.0, 0.6, 1.3, 0.0, 0.0], dtype=jnp.float64)
        sigma_array = jnp.ones(5, dtype=jnp.float64)
        b1_array = jnp.full(5, -0.3, dtype=jnp.float64)
        b2_array = jnp.full(5, 0.3, dtype=jnp.float64)

        expected = float(
            compute_heterog_multistage_logfptd(
                1.9,
                1,
                0.0,
                1.5,
                -1.5,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                3,
                order=30,
                trunc_num=25,
                log_space=True,
            )
        )
        actual = float(
            compute_heterog_multistage_logfptd_jit(
                1.9,
                1,
                0.0,
                1.5,
                -1.5,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                3,
                order=30,
                trunc_num=25,
                log_space=True,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
