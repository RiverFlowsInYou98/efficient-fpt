"""Log-space regression tests for Python, JAX, and Cython backends."""

import warnings

import numpy as np
import pytest

from efficient_fpt.multi_stage import compute_homog_multistage_fptds_and_npd

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from efficient_fpt.jax.multi_stage import (
    compute_addm_fptd as compute_addm_fptd_jax,
    compute_heterog_multistage_fptd as compute_heterog_multistage_fptd_jax,
)


from helpers import try_import_cython_multi_stage as _try_import_cython


class TestLogSpaceAgreement:
    RTOL = 1e-5
    ATOL = 1e-10

    def test_python_homog_multistage_logspace_matches_normal(self):
        t_grid = np.array([2.3], dtype=np.float64)
        node_array = np.array([0.0, 0.8, 1.6], dtype=np.float64)
        mu_array = np.array([0.5, -0.3, 0.2], dtype=np.float64)
        sigma_array = np.ones(3, dtype=np.float64)
        b1_array = np.full(3, -0.3, dtype=np.float64)
        b2_array = np.full(3, 0.3, dtype=np.float64)
        x0 = np.array([[1.0], [0.0]], dtype=np.float64)

        result_normal, _ = compute_homog_multistage_fptds_and_npd(
            t_grid,
            3.0,
            x0,
            1.5,
            -1.5,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            trunc_num=50,
            threshold=1e-30,
            adaptive_stopping=False,
            log_space=False,
        )
        result_log, _ = compute_homog_multistage_fptds_and_npd(
            t_grid,
            3.0,
            x0,
            1.5,
            -1.5,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            trunc_num=50,
            threshold=1e-30,
            adaptive_stopping=False,
            log_space=True,
        )

        np.testing.assert_allclose(
            result_log, result_normal, rtol=self.RTOL, atol=self.ATOL
        )

    @pytest.mark.parametrize("choice", [1, -1])
    def test_jax_heterog_multistage_logspace_matches_normal(self, choice):
        mu_array = jnp.array([0.5, -0.3, 0.2, 0.0, 0.0], dtype=jnp.float64)
        node_array = jnp.array([0.0, 0.8, 1.6, 0.0, 0.0], dtype=jnp.float64)
        sigma_array = jnp.ones(5, dtype=jnp.float64)
        b1_array = jnp.full(5, -0.3, dtype=jnp.float64)
        b2_array = jnp.full(5, 0.3, dtype=jnp.float64)

        normal = float(
            compute_heterog_multistage_fptd_jax(
                2.3,
                choice,
                0.0,
                1.5,
                -1.5,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                3,
                trunc_num=50,
                log_space=False,
            )
        )
        log_val = float(
            compute_heterog_multistage_fptd_jax(
                2.3,
                choice,
                0.0,
                1.5,
                -1.5,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                3,
                trunc_num=50,
                log_space=True,
            )
        )

        assert normal > 0.0
        assert log_val > 0.0
        np.testing.assert_allclose(log_val, normal, rtol=self.RTOL, atol=self.ATOL)

    def test_jax_addm_logspace_matches_normal(self):
        sacc_array = jnp.array([0.0, 0.8, 1.6, 0.0], dtype=jnp.float64)

        normal = float(
            compute_addm_fptd_jax(
                2.3,
                1,
                0.0,
                1.0,
                1.0,
                1.5,
                0.3,
                0.0,
                0.5,
                0.3,
                0,
                sacc_array,
                3,
                trunc_num=50,
                log_space=False,
            )
        )
        log_val = float(
            compute_addm_fptd_jax(
                2.3,
                1,
                0.0,
                1.0,
                1.0,
                1.5,
                0.3,
                0.0,
                0.5,
                0.3,
                0,
                sacc_array,
                3,
                trunc_num=50,
                log_space=True,
            )
        )

        assert normal > 0.0
        assert log_val > 0.0
        np.testing.assert_allclose(log_val, normal, rtol=self.RTOL, atol=self.ATOL)

    @pytest.mark.parametrize("choice", [1, -1])
    def test_cython_heterog_multistage_logspace_matches_normal(self, choice):
        _, compute_heterog_multistage_fptd_cy = _try_import_cython()
        if compute_heterog_multistage_fptd_cy is None:
            pytest.skip("Cython implementation not available")

        mu_array = np.array([0.5, -0.3, 0.2], dtype=np.float64)
        node_array = np.array([0.0, 0.8, 1.6], dtype=np.float64)
        sigma_array = np.ones(3, dtype=np.float64)
        b1_array = np.full(3, -0.3, dtype=np.float64)
        b2_array = np.full(3, 0.3, dtype=np.float64)

        normal = compute_heterog_multistage_fptd_cy(
            2.3,
            choice,
            0.0,
            1.5,
            -1.5,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            3,
            trunc_num=50,
            threshold=1e-30,
            log_space=False,
        )
        log_val = compute_heterog_multistage_fptd_cy(
            2.3,
            choice,
            0.0,
            1.5,
            -1.5,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            3,
            trunc_num=50,
            threshold=1e-30,
            log_space=True,
        )

        assert normal > 0.0
        assert log_val > 0.0
        np.testing.assert_allclose(log_val, normal, rtol=1e-8, atol=1e-12)


class TestLogSpaceZeroMass:
    def test_python_zero_mass_is_exact_and_warning_free(self):
        t_grid = np.array([3.1], dtype=np.float64)
        node_array = np.array([0.0, 1.0], dtype=np.float64)
        mu_array = np.array([0.2, -0.1], dtype=np.float64)
        sigma_array = np.ones(2, dtype=np.float64)
        b1_array = np.full(2, -0.5, dtype=np.float64)
        b2_array = np.full(2, 0.5, dtype=np.float64)
        x0 = np.array([[1.0], [0.0]], dtype=np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result_normal, _ = compute_homog_multistage_fptds_and_npd(
                t_grid,
                3.5,
                x0,
                1.0,
                -1.0,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                trunc_num=50,
                adaptive_stopping=False,
                log_space=False,
            )
            result_log, _ = compute_homog_multistage_fptds_and_npd(
                t_grid,
                3.5,
                x0,
                1.0,
                -1.0,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                trunc_num=50,
                adaptive_stopping=False,
                log_space=True,
            )

        assert len(caught) == 0
        assert result_normal[1, 0] == 0.0
        assert result_normal[2, 0] == 0.0
        assert result_log[1, 0] == 0.0
        assert result_log[2, 0] == 0.0

    def test_cython_zero_mass_is_exact(self):
        compute_addm_fptd_cy, compute_heterog_multistage_fptd_cy = _try_import_cython()
        if compute_addm_fptd_cy is None or compute_heterog_multistage_fptd_cy is None:
            pytest.skip("Cython implementation not available")

        mu_array = np.array([0.2, -0.1], dtype=np.float64)
        node_array = np.array([0.0, 1.0], dtype=np.float64)
        sigma_array = np.ones(2, dtype=np.float64)
        b1_array = np.full(2, -0.5, dtype=np.float64)
        b2_array = np.full(2, 0.5, dtype=np.float64)
        sacc_array = np.array([0.0, 1.0], dtype=np.float64)

        generalized_normal = compute_heterog_multistage_fptd_cy(
            3.1,
            1,
            0.0,
            1.0,
            -1.0,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            2,
            trunc_num=50,
            log_space=False,
        )
        generalized_log = compute_heterog_multistage_fptd_cy(
            3.1,
            1,
            0.0,
            1.0,
            -1.0,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            2,
            trunc_num=50,
            log_space=True,
        )
        addm_normal = compute_addm_fptd_cy(
            3.1,
            1,
            0.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.0,
            0.2,
            0.1,
            0,
            sacc_array,
            2,
            trunc_num=50,
            log_space=False,
        )
        addm_log = compute_addm_fptd_cy(
            3.1,
            1,
            0.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.0,
            0.2,
            0.1,
            0,
            sacc_array,
            2,
            trunc_num=50,
            log_space=True,
        )

        assert generalized_normal == 0.0
        assert generalized_log == 0.0
        assert addm_normal == 0.0
        assert addm_log == 0.0

    def test_jax_zero_mass_is_exact(self):
        mu_array = jnp.array([0.2, -0.1, 0.0, 0.0], dtype=jnp.float64)
        node_array = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
        sigma_array = jnp.ones(4, dtype=jnp.float64)
        b1_array = jnp.full(4, -0.5, dtype=jnp.float64)
        b2_array = jnp.full(4, 0.5, dtype=jnp.float64)
        sacc_array = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

        generalized_normal = float(
            compute_heterog_multistage_fptd_jax(
                3.1,
                1,
                0.0,
                1.0,
                -1.0,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                2,
                trunc_num=50,
                log_space=False,
            )
        )
        generalized_log = float(
            compute_heterog_multistage_fptd_jax(
                3.1,
                1,
                0.0,
                1.0,
                -1.0,
                mu_array,
                node_array,
                sigma_array,
                b1_array,
                b2_array,
                2,
                trunc_num=50,
                log_space=True,
            )
        )
        addm_normal = float(
            compute_addm_fptd_jax(
                3.1,
                1,
                0.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.0,
                0.2,
                0.1,
                0,
                sacc_array,
                2,
                trunc_num=50,
                log_space=False,
            )
        )
        addm_log = float(
            compute_addm_fptd_jax(
                3.1,
                1,
                0.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.0,
                0.2,
                0.1,
                0,
                sacc_array,
                2,
                trunc_num=50,
                log_space=True,
            )
        )

        assert generalized_normal == 0.0
        assert generalized_log == 0.0
        assert addm_normal == 0.0
        assert addm_log == 0.0
