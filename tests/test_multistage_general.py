import numpy as np
import pytest

from efficient_fpt.multi_stage import compute_homog_multistage_fptds_and_npd

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from efficient_fpt.jax.multi_stage import (
    compute_addm_fptd as compute_addm_fptd_jax,
    compute_heterog_multistage_fptd as compute_heterog_multistage_fptd_jax,
    compute_heterog_multistage_fptd_stagescan,
)


from helpers import try_import_cython_multi_stage as _try_import_cython


def _python_single_trial_fptd(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    order=30,
    trunc_num=100,
):
    t_grid = np.array([rt], dtype=np.float64)
    result, _ = compute_homog_multistage_fptds_and_npd(
        t_grid,
        T=rt + 1.0,
        x0=np.array([[1.0], [x0]], dtype=np.float64),
        a1=a1,
        a2=a2,
        mu_array=mu_array,
        node_array=node_array,
        sigma_array=sigma_array,
        b1_array=b1_array,
        b2_array=b2_array,
        order=order,
        trunc_num=trunc_num,
        threshold=1e-30,
        adaptive_stopping=False,
    )
    row = 1 if choice == 1 else 2
    return result[row, 0] if result.shape[1] else 0.0


def _addm_public_args(rt, choice, sigma, a, b, x0, mu1, mu2, node_array, d):
    return dict(
        rt=rt,
        choice=choice,
        eta=0.0,
        kappa=1.0,
        sigma=sigma,
        a=a,
        b=b,
        x0=x0,
        r1=mu1,
        r2=-mu2,
        flag=0,
        sacc_array=jnp.array(node_array, dtype=jnp.float64),
        d=d,
    )


class TestSymmetricEquivalence:
    RTOL = 1e-6
    ATOL = 1e-10

    @pytest.mark.parametrize("choice", [1, -1])
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_jax_heterog_matches_addm_for_symmetric_case(self, d, choice):
        sigma, a, b, x0 = 1.0, 1.5, 0.3, 0.0
        mu1, mu2 = 0.5, -0.3
        node_array = np.arange(d, dtype=np.float64)
        mu_array = np.where(np.arange(d) % 2 == 0, mu1, mu2)
        sigma_array = np.full(d, sigma, dtype=np.float64)
        b1_array = np.full(d, -b, dtype=np.float64)
        b2_array = np.full(d, b, dtype=np.float64)
        rt = node_array[d - 1] + 0.5

        addm_result = float(
            compute_addm_fptd_jax(
                **_addm_public_args(
                    rt, choice, sigma, a, b, x0, mu1, mu2, node_array, d
                ),
                trunc_num=50,
            )
        )
        heterog_result = float(
            compute_heterog_multistage_fptd_jax(
                rt,
                choice,
                x0,
                a,
                -a,
                jnp.array(mu_array, dtype=jnp.float64),
                jnp.array(node_array, dtype=jnp.float64),
                jnp.array(sigma_array, dtype=jnp.float64),
                jnp.array(b1_array, dtype=jnp.float64),
                jnp.array(b2_array, dtype=jnp.float64),
                d,
                trunc_num=50,
            )
        )

        np.testing.assert_allclose(
            addm_result, heterog_result, rtol=self.RTOL, atol=self.ATOL
        )

    @pytest.mark.parametrize("choice", [1, -1])
    def test_jax_precomputed_matches_stagescan(self, choice):
        d = 3
        max_d = 5
        mu_array = jnp.array([0.4, -0.2, 0.4, 0.0, 0.0], dtype=jnp.float64)
        node_array = jnp.array([0.0, 0.9, 1.8, 0.0, 0.0], dtype=jnp.float64)
        sigma_array = jnp.ones(max_d, dtype=jnp.float64)
        b1_array = jnp.full(max_d, -0.3, dtype=jnp.float64)
        b2_array = jnp.full(max_d, 0.3, dtype=jnp.float64)
        rt = 2.2

        slow = float(
            compute_heterog_multistage_fptd_stagescan(
                rt,
                choice,
                0.0,
                1.5,
                -1.5,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                d,
                trunc_num=50,
            )
        )
        fast = float(
            compute_heterog_multistage_fptd_jax(
                rt,
                choice,
                0.0,
                1.5,
                -1.5,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                d,
                trunc_num=50,
            )
        )

        np.testing.assert_allclose(slow, fast, rtol=1e-6, atol=1e-10)


class TestCrossBackendEquivalence:
    RTOL = 1e-4
    ATOL = 1e-8

    @pytest.mark.parametrize("choice", [1, -1])
    def test_asymmetric_cross_backend(self, choice):
        d = 3
        mu_array = np.array([0.4, -0.2, 0.1], dtype=np.float64)
        node_array = np.array([0.0, 0.8, 1.6], dtype=np.float64)
        sigma_array = np.array([0.9, 1.1, 1.0], dtype=np.float64)
        b1_array = np.array([-0.4, -0.3, -0.2], dtype=np.float64)
        b2_array = np.array([0.2, 0.25, 0.3], dtype=np.float64)
        a1, a2, x0, rt = 1.8, -1.2, 0.0, 2.1

        py_result = _python_single_trial_fptd(
            rt,
            choice,
            x0,
            a1,
            a2,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
        )

        max_d = 5
        jax_result = float(
            compute_heterog_multistage_fptd_jax(
                rt,
                choice,
                x0,
                a1,
                a2,
                jnp.array(np.pad(mu_array, (0, max_d - d)), dtype=jnp.float64),
                jnp.array(np.pad(node_array, (0, max_d - d)), dtype=jnp.float64),
                jnp.array(
                    np.pad(
                        sigma_array, (0, max_d - d), constant_values=sigma_array[-1]
                    ),
                    dtype=jnp.float64,
                ),
                jnp.array(
                    np.pad(b1_array, (0, max_d - d), constant_values=b1_array[-1]),
                    dtype=jnp.float64,
                ),
                jnp.array(
                    np.pad(b2_array, (0, max_d - d), constant_values=b2_array[-1]),
                    dtype=jnp.float64,
                ),
                d,
                trunc_num=100,
            )
        )

        np.testing.assert_allclose(
            py_result, jax_result, rtol=self.RTOL, atol=self.ATOL
        )

        _, compute_heterog_multistage_fptd_cy = _try_import_cython()
        if compute_heterog_multistage_fptd_cy is not None:
            cy_result = compute_heterog_multistage_fptd_cy(
                rt,
                choice,
                x0,
                a1,
                a2,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                d,
                trunc_num=100,
                threshold=1e-30,
            )
            np.testing.assert_allclose(
                py_result, cy_result, rtol=self.RTOL, atol=self.ATOL
            )


class TestQuadratureOrder:
    def test_addm_order_parameter_supported(self):
        args = _addm_public_args(
            rt=2.4,
            choice=1,
            sigma=1.0,
            a=1.5,
            b=0.3,
            x0=0.0,
            mu1=0.5,
            mu2=-0.3,
            node_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            d=3,
        )

        result_20 = float(compute_addm_fptd_jax(**args, order=20, trunc_num=50))
        result_30 = float(compute_addm_fptd_jax(**args, order=30, trunc_num=50))
        assert result_20 > 0
        assert result_30 > 0
        np.testing.assert_allclose(result_20, result_30, rtol=1e-4)
