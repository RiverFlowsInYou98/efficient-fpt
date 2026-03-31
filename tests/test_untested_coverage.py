"""Tests for previously-untested functionality (audit bug #16).

Covers:
- boundaries.py: piecewise_const_func, piecewise_linear_func, weibull_survival
- utils.py: adaptive_interpolation
- models.py: MultiStageModel construction and methods
- models.py: aDDModel.mean_neg_log_likelihood
- jax/single_stage.py: log_fptd_basic, log_q_basic, log_fptd_single, log_q_single
- jax/utils.py: lgwt_lookup_table
- cython/batch.pyx: compute_tada_mean_nll
"""

import numpy as np
import pytest

from efficient_fpt.boundaries import (
    piecewise_const_func,
    piecewise_linear_func,
    weibull_survival,
)
from efficient_fpt.utils import adaptive_interpolation


# ---------------------------------------------------------------------------
# boundaries.py
# ---------------------------------------------------------------------------


class TestPiecewiseConstFunc:
    def test_single_stage(self):
        mu = np.array([2.0])
        nodes = np.array([0.0])
        assert piecewise_const_func(0.5, mu, nodes) == 2.0

    def test_multi_stage(self):
        mu = np.array([1.0, 3.0, 5.0])
        nodes = np.array([0.0, 1.0, 2.0])
        assert piecewise_const_func(0.5, mu, nodes) == 1.0
        assert piecewise_const_func(1.5, mu, nodes) == 3.0
        assert piecewise_const_func(2.5, mu, nodes) == 5.0

    def test_array_input(self):
        mu = np.array([1.0, 3.0])
        nodes = np.array([0.0, 1.0])
        t = np.array([0.2, 0.8, 1.2, 1.8])
        result = piecewise_const_func(t, mu, nodes)
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0, 3.0])


class TestPiecewiseLinearFunc:
    def test_flat_segments(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        nodes = np.array([0.0, 1.0])
        assert piecewise_linear_func(0.5, a, b, nodes) == 1.0
        assert piecewise_linear_func(1.5, a, b, nodes) == 2.0

    def test_sloped_segments(self):
        a = np.array([1.0])
        b = np.array([-0.5])
        nodes = np.array([0.0])
        np.testing.assert_allclose(
            piecewise_linear_func(2.0, a, b, nodes), 0.0
        )

    def test_array_input(self):
        a = np.array([1.0, 0.5])
        b = np.array([-0.5, 0.0])
        nodes = np.array([0.0, 1.0])
        t = np.array([0.0, 0.5, 1.0, 1.5])
        result = piecewise_linear_func(t, a, b, nodes)
        np.testing.assert_allclose(result, [1.0, 0.75, 0.5, 0.5])


class TestWeibullSurvival:
    def test_at_zero(self):
        assert weibull_survival(t=0, lbda=1, k=1) == 1.0

    def test_exponential_decay(self):
        # k=1 gives exponential: exp(-t/lambda)
        result = weibull_survival(t=1.0, lbda=1.0, k=1)
        np.testing.assert_allclose(result, np.exp(-1.0))

    def test_array_input(self):
        t = np.array([0, 1, 2])
        result = weibull_survival(t=t, lbda=1, k=2)
        expected = np.exp(-t**2)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# utils.py: adaptive_interpolation
# ---------------------------------------------------------------------------


class TestAdaptiveInterpolation:
    def test_linear_function_exact(self):
        f = lambda x: 2 * x + 1
        x_pts, y_pts = adaptive_interpolation(f, (0, 10), error_threshold=1e-10)
        xi = np.linspace(0, 10, 200)
        yi = np.interp(xi, x_pts, y_pts)
        np.testing.assert_allclose(yi, f(xi), atol=1e-10)

    def test_quadratic_refines(self):
        f = lambda x: x**2
        x_pts, y_pts = adaptive_interpolation(
            f, (0, 1), error_threshold=1e-4, initial_points=3
        )
        # Should have added points beyond the initial 3
        assert len(x_pts) > 3
        xi = np.linspace(0, 1, 500)
        yi = np.interp(xi, x_pts, y_pts)
        np.testing.assert_allclose(yi, f(xi), atol=1e-4)

    def test_max_iterations_warning(self):
        f = lambda x: np.sin(100 * x)
        with pytest.warns(RuntimeWarning, match="Maximum iterations"):
            adaptive_interpolation(
                f, (0, 1), error_threshold=1e-15, max_iterations=5
            )


# ---------------------------------------------------------------------------
# models.py: MultiStageModel
# ---------------------------------------------------------------------------


class TestMultiStageModel:
    def test_construction(self):
        from efficient_fpt.models import MultiStageModel

        mu = np.array([1.0, 2.0])
        nodes = np.array([0.0, 0.5])
        sigma = np.array([1.0, 1.0])
        b1 = np.array([-0.5, -0.5])
        b2 = np.array([0.5, 0.5])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)
        assert model.d == 2
        np.testing.assert_array_equal(model.mu_array, mu)

    def test_drift_and_diffusion(self):
        from efficient_fpt.models import MultiStageModel

        mu = np.array([1.0, 3.0])
        nodes = np.array([0.0, 1.0])
        sigma = np.array([0.5, 1.5])
        b1 = np.array([0.0, 0.0])
        b2 = np.array([0.0, 0.0])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)

        assert model.drift_coeff(0, 0.5) == 1.0
        assert model.drift_coeff(0, 1.5) == 3.0
        assert model.diffusion_coeff(0, 0.5) == 0.5
        assert model.diffusion_coeff(0, 1.5) == 1.5

    def test_boundaries(self):
        from efficient_fpt.models import MultiStageModel

        mu = np.array([1.0])
        nodes = np.array([0.0])
        sigma = np.array([1.0])
        b1 = np.array([-0.5])
        b2 = np.array([0.5])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)

        np.testing.assert_allclose(model.upper_bdy(0.0), 1.0)
        np.testing.assert_allclose(model.upper_bdy(1.0), 0.5)
        np.testing.assert_allclose(model.lower_bdy(0.0), -1.0)
        np.testing.assert_allclose(model.lower_bdy(1.0), -0.5)

    def test_is_update_vectorizable(self):
        from efficient_fpt.models import MultiStageModel

        mu = np.array([1.0])
        nodes = np.array([0.0])
        sigma = np.array([1.0])
        b1 = np.array([0.0])
        b2 = np.array([0.0])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)
        assert model.is_update_vectorizable is True

    def test_boundary_intercepts_propagate(self):
        from efficient_fpt.models import MultiStageModel

        mu = np.array([1.0, 2.0])
        nodes = np.array([0.0, 1.0])
        sigma = np.array([1.0, 1.0])
        b1 = np.array([-0.2, -0.3])
        b2 = np.array([0.2, 0.3])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)
        # At t=0: ub=1.0, lb=-1.0
        # Stage 0 spans [0, 1) with slope b1=-0.2, b2=0.2
        # At t=1: ub = 1.0 + (-0.2)*1 = 0.8, lb = -1.0 + 0.2*1 = -0.8
        np.testing.assert_allclose(model.ub_array, [1.0, 0.8])
        np.testing.assert_allclose(model.lb_array, [-1.0, -0.8])


# ---------------------------------------------------------------------------
# models.py: aDDModel.mean_neg_log_likelihood
# ---------------------------------------------------------------------------


class TestADDModelMeanNLL:
    @pytest.fixture
    def addm_data(self):
        from efficient_fpt.models import aDDModel

        model = aDDModel(eta=0.5, kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0)
        data = model.generate_experiment(n_trials=50, rng=42)
        return model, data

    def test_returns_finite_scalar(self, addm_data):
        model, data = addm_data
        cov = data["covariates"]
        nll = model.mean_neg_log_likelihood(
            rt_data=data["decision_data"]["rt_data"],
            choice_data=data["decision_data"]["choice_data"],
            r1_data=np.asarray(cov["r1_data"], dtype=np.float64),
            r2_data=np.asarray(cov["r2_data"], dtype=np.float64),
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        assert np.isfinite(nll)
        assert isinstance(nll, float)

    def test_different_params_change_nll(self, addm_data):
        model, data = addm_data
        cov = data["covariates"]
        kwargs = dict(
            rt_data=data["decision_data"]["rt_data"],
            choice_data=data["decision_data"]["choice_data"],
            r1_data=np.asarray(cov["r1_data"], dtype=np.float64),
            r2_data=np.asarray(cov["r2_data"], dtype=np.float64),
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        nll1 = model.mean_neg_log_likelihood(**kwargs)
        # Change eta and verify NLL changes
        from efficient_fpt.models import aDDModel
        model2 = aDDModel(eta=0.9, kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0)
        nll2 = model2.mean_neg_log_likelihood(**kwargs)
        assert nll1 != nll2


# ---------------------------------------------------------------------------
# JAX: log_fptd_basic, log_q_basic, log_fptd_single, log_q_single
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestJAXLogFunctions:
    @pytest.fixture(autouse=True)
    def _import_jax(self):
        from efficient_fpt.jax.single_stage import (
            fptd_basic,
            q_basic,
            fptd_single,
            q_single,
            log_fptd_basic,
            log_q_basic,
            log_fptd_single,
            log_q_single,
        )
        self.fptd_basic = fptd_basic
        self.q_basic = q_basic
        self.fptd_single = fptd_single
        self.q_single = q_single
        self.log_fptd_basic = log_fptd_basic
        self.log_q_basic = log_q_basic
        self.log_fptd_single = log_fptd_single
        self.log_q_single = log_q_single

    def test_log_fptd_basic_matches_log_of_fptd(self):
        t = 0.5
        mu, a1, b1, a2, b2 = 1.0, 1.0, -0.5, -1.0, 0.5
        for bdy in [1, -1]:
            density = self.fptd_basic(t, mu, a1, b1, a2, b2, bdy)
            log_density = self.log_fptd_basic(t, mu, a1, b1, a2, b2, bdy)
            np.testing.assert_allclose(
                float(log_density), float(jnp.log(density)), rtol=1e-10
            )

    def test_log_q_basic_matches_log_of_q(self):
        x = 0.3
        mu, a1, b1, a2, b2, T = 1.0, 1.0, -0.5, -1.0, 0.5, 0.5
        density = self.q_basic(x, mu, a1, b1, a2, b2, T)
        log_density = self.log_q_basic(x, mu, a1, b1, a2, b2, T)
        np.testing.assert_allclose(
            float(log_density), float(jnp.log(density)), rtol=1e-10
        )

    def test_log_fptd_single_matches_log_of_fptd(self):
        t = 0.5
        mu, sigma, a1, b1, a2, b2, x0 = 1.0, 1.5, 1.0, -0.5, -1.0, 0.5, 0.1
        for bdy in [1, -1]:
            density = self.fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy)
            log_density = self.log_fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy)
            np.testing.assert_allclose(
                float(log_density), float(jnp.log(density)), rtol=1e-10
            )

    def test_log_q_single_matches_log_of_q(self):
        x = 0.3
        mu, sigma, a1, b1, a2, b2, T, x0 = 1.0, 1.5, 1.0, -0.5, -1.0, 0.5, 0.5, 0.1
        density = self.q_single(x, mu, sigma, a1, b1, a2, b2, T, x0)
        log_density = self.log_q_single(x, mu, sigma, a1, b1, a2, b2, T, x0)
        np.testing.assert_allclose(
            float(log_density), float(jnp.log(density)), rtol=1e-10
        )

    def test_log_fptd_basic_array(self):
        t = jnp.array([0.3, 0.5, 0.8])
        mu, a1, b1, a2, b2 = 1.0, 1.0, -0.5, -1.0, 0.5
        log_density = self.log_fptd_basic(t, mu, a1, b1, a2, b2, 1)
        assert log_density.shape == (3,)
        assert jnp.all(jnp.isfinite(log_density))

    def test_log_returns_neg_inf_for_zero_density(self):
        # At t=0, density should be 0, log should be -inf
        log_d = self.log_fptd_basic(0.0, 1.0, 1.0, -0.5, -1.0, 0.5, 1)
        assert float(log_d) == float("-inf")


# ---------------------------------------------------------------------------
# JAX: lgwt_lookup_table
# ---------------------------------------------------------------------------


class TestLgwtLookupTable:
    def test_correct_length(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        x, w = lgwt_lookup_table(10, 0.0, 1.0)
        assert x.shape == (10,)
        assert w.shape == (10,)

    def test_nodes_in_interval(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        x, w = lgwt_lookup_table(20, -2.0, 3.0)
        assert float(jnp.min(x)) >= -2.0
        assert float(jnp.max(x)) <= 3.0

    def test_weights_sum_to_interval_length(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        x, w = lgwt_lookup_table(30, 2.0, 5.0)
        np.testing.assert_allclose(float(jnp.sum(w)), 3.0, rtol=1e-12)

    def test_integrates_polynomial_exactly(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        # Order-n Gauss-Legendre integrates polynomials up to degree 2n-1 exactly
        x, w = lgwt_lookup_table(5, 0.0, 1.0)
        # Integral of x^4 from 0 to 1 = 1/5 (degree 4 < 2*5-1=9)
        result = float(jnp.sum(w * x**4))
        np.testing.assert_allclose(result, 0.2, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cython: compute_tada_mean_nll
# ---------------------------------------------------------------------------


class TestTADAMeanNLL:
    @pytest.fixture
    def tada_data(self):
        from efficient_fpt.models import aDDModel

        model = aDDModel(eta=0.5, kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0)
        return model.generate_experiment(n_trials=30, rng=123)

    def test_returns_finite_scalar(self, tada_data):
        try:
            from efficient_fpt.cython.batch import compute_tada_mean_nll
        except ImportError:
            pytest.skip("Cython extension not available")

        cov = tada_data["covariates"]
        nll = compute_tada_mean_nll(
            rt_data=tada_data["decision_data"]["rt_data"],
            choice_data=tada_data["decision_data"]["choice_data"],
            eta=0.5,
            kappa=1.0,
            sigma=1.0,
            a=1.0,
            b=0.5,
            x0=0.0,
            r1_data=np.asarray(cov["r1_data"], dtype=np.float64),
            r2_data=np.asarray(cov["r2_data"], dtype=np.float64),
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        assert np.isfinite(nll)
        assert isinstance(nll, float)

    def test_different_params_change_nll(self, tada_data):
        try:
            from efficient_fpt.cython.batch import compute_tada_mean_nll
        except ImportError:
            pytest.skip("Cython extension not available")

        cov = tada_data["covariates"]
        common = dict(
            rt_data=tada_data["decision_data"]["rt_data"],
            choice_data=tada_data["decision_data"]["choice_data"],
            kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0,
            r1_data=np.asarray(cov["r1_data"], dtype=np.float64),
            r2_data=np.asarray(cov["r2_data"], dtype=np.float64),
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        nll1 = compute_tada_mean_nll(eta=0.5, **common)
        nll2 = compute_tada_mean_nll(eta=0.9, **common)
        assert nll1 != nll2
