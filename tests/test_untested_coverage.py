"""Tests for previously-untested functionality (audit bug #16).

Covers:
- boundaries.py: piecewise_const_func, piecewise_linear_func, weibull_survival
- utils.py: adaptive_interpolation
- models.py: MultiStageModel construction and methods
- models.py: aDDModel.mean_neg_log_likelihood
- cython/batch.pyx: compute_tada_mean_nll
"""

import numpy as np
import pytest

from efpt.boundaries import (
    piecewise_const_func,
    piecewise_linear_func,
    weibull_survival,
)
from efpt.utils import adaptive_interpolation


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
        from efpt.models import MultiStageModel

        mu = np.array([1.0, 2.0])
        nodes = np.array([0.0, 0.5])
        sigma = np.array([1.0, 1.0])
        b1 = np.array([-0.5, -0.5])
        b2 = np.array([0.5, 0.5])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)
        assert model.d == 2
        np.testing.assert_array_equal(model.mu_array, mu)

    def test_drift_and_diffusion(self):
        from efpt.models import MultiStageModel

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
        from efpt.models import MultiStageModel

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
        from efpt.models import MultiStageModel

        mu = np.array([1.0])
        nodes = np.array([0.0])
        sigma = np.array([1.0])
        b1 = np.array([0.0])
        b2 = np.array([0.0])
        model = MultiStageModel(mu, nodes, sigma, 1.0, b1, -1.0, b2, 0.0)
        assert model.is_update_vectorizable is True

    def test_boundary_intercepts_propagate(self):
        from efpt.models import MultiStageModel

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
    def test_returns_finite_scalar(self, addm_model, addm_experiment):
        cov = addm_experiment["covariates"]
        nll = addm_model.mean_neg_log_likelihood(
            rt_data=addm_experiment["decision_data"]["rt_data"],
            choice_data=addm_experiment["decision_data"]["choice_data"],
            r1_data=cov["r1_data"],
            r2_data=cov["r2_data"],
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        assert np.isfinite(nll)
        assert isinstance(nll, float)

    def test_different_params_change_nll(self, addm_model, addm_experiment):
        cov = addm_experiment["covariates"]
        kwargs = dict(
            rt_data=addm_experiment["decision_data"]["rt_data"],
            choice_data=addm_experiment["decision_data"]["choice_data"],
            r1_data=cov["r1_data"],
            r2_data=cov["r2_data"],
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        nll1 = addm_model.mean_neg_log_likelihood(**kwargs)
        # Change eta and verify NLL changes
        from efpt.models import aDDModel
        model2 = aDDModel(eta=0.9, kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0)
        nll2 = model2.mean_neg_log_likelihood(**kwargs)
        assert nll1 != nll2


# ---------------------------------------------------------------------------
# Cython: compute_tada_mean_nll
# ---------------------------------------------------------------------------


class TestTADAMeanNLL:
    def test_returns_finite_scalar(self, addm_experiment):
        try:
            from efpt.cython.batch import compute_tada_mean_nll
        except ImportError:
            pytest.skip("Cython extension not available")

        cov = addm_experiment["covariates"]
        nll = compute_tada_mean_nll(
            rt_data=addm_experiment["decision_data"]["rt_data"],
            choice_data=addm_experiment["decision_data"]["choice_data"],
            eta=0.5,
            kappa=1.0,
            sigma=1.0,
            a=1.0,
            b=0.5,
            x0=0.0,
            r1_data=cov["r1_data"],
            r2_data=cov["r2_data"],
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        assert np.isfinite(nll)
        assert isinstance(nll, float)

    def test_different_params_change_nll(self, addm_experiment):
        try:
            from efpt.cython.batch import compute_tada_mean_nll
        except ImportError:
            pytest.skip("Cython extension not available")

        cov = addm_experiment["covariates"]
        common = dict(
            rt_data=addm_experiment["decision_data"]["rt_data"],
            choice_data=addm_experiment["decision_data"]["choice_data"],
            kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0,
            r1_data=cov["r1_data"],
            r2_data=cov["r2_data"],
            flag_data=cov["flag_data"],
            sacc_array_data=cov["sacc_array_data"],
            d_data=cov["d_data"],
        )
        nll1 = compute_tada_mean_nll(eta=0.5, **common)
        nll2 = compute_tada_mean_nll(eta=0.9, **common)
        assert nll1 != nll2
