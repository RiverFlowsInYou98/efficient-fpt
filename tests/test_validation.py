"""Tests for input validation."""

import numpy as np
import pytest

from efpt.validation import check_addm_params, check_multistage_params
from efpt.models import SingleStageModel, aDDModel, MultiStageModel


class TestValidateAddmParams:
    """Tests for check_addm_params."""

    def test_valid_params_pass(self):
        check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=1.5, b=0.3, x0=0.0)

    def test_zero_sigma_allowed(self):
        check_addm_params(eta=0.5, kappa=1.0, sigma=0.0, a=1.5, b=0.0, x0=0.0)

    def test_infinite_a_allowed(self):
        check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=np.inf, b=0.0, x0=0.0)

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=-1.0, a=1.5, b=0.3, x0=0.0)

    def test_nan_sigma_rejected(self):
        with pytest.raises(ValueError, match="sigma must be finite"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=np.nan, a=1.5, b=0.3, x0=0.0)

    def test_zero_a_rejected(self):
        with pytest.raises(ValueError, match="a.*must be positive"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=0.0, b=0.3, x0=0.0)

    def test_negative_a_rejected(self):
        with pytest.raises(ValueError, match="a.*must be positive"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=-1.0, b=0.3, x0=0.0)

    def test_negative_b_rejected(self):
        with pytest.raises(ValueError, match="b.*must be non-negative"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=1.5, b=-0.1, x0=0.0)

    def test_x0_at_upper_boundary_rejected_when_collapsing(self):
        with pytest.raises(ValueError, match="x0 must be within"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=1.5, b=0.3, x0=1.5)

    def test_x0_at_lower_boundary_rejected_when_collapsing(self):
        with pytest.raises(ValueError, match="x0 must be within"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=1.5, b=0.3, x0=-1.5)

    def test_x0_at_boundary_rejected_when_not_collapsing(self):
        with pytest.raises(ValueError, match="x0 must be within"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=1.5, b=0.0, x0=1.5)

    def test_infinite_x0_rejected_even_with_infinite_boundary(self):
        with pytest.raises(ValueError, match="x0 must be finite"):
            check_addm_params(eta=0.5, kappa=1.0, sigma=1.0, a=np.inf, b=0.0, x0=np.inf)


class TestAddmModelValidation:
    """Test that aDDModel.__init__ calls validation."""

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            aDDModel(eta=0.5, kappa=1.0, sigma=-1.0, a=1.5, b=0.3, x0=0.0)

    def test_invalid_a_raises(self):
        with pytest.raises(ValueError, match="a.*must be positive"):
            aDDModel(eta=0.5, kappa=1.0, sigma=1.0, a=0.0, b=0.3, x0=0.0)

    def test_infinite_boundary_is_allowed(self):
        model = aDDModel(eta=0.5, kappa=1.0, sigma=1.0, a=np.inf, b=0.0, x0=0.0)
        assert np.isposinf(model.a)


class TestSingleStageModelValidation:
    """Tests for SingleStageModel constructor validation."""

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            SingleStageModel(mu=0.1, sigma=-1.0, a=1.5, b=0.2, x0=0.0)

    def test_x0_outside_finite_boundaries_rejected(self):
        with pytest.raises(ValueError, match="x0 must be within"):
            SingleStageModel(mu=0.1, sigma=1.0, a=1.5, b=0.0, x0=2.0)

    def test_infinite_boundary_is_allowed(self):
        model = SingleStageModel(mu=0.1, sigma=1.0, a=np.inf, b=0.0, x0=0.0)
        assert np.isposinf(model.a)


class TestMultiStageValidation:
    """Tests for check_multistage_params."""

    def test_valid_params(self):
        check_multistage_params(
            mu_array=np.array([1.0, 2.0]),
            node_array=np.array([0.0, 0.5]),
            sigma_array=np.array([1.0, 1.0]),
            a1=1.5,
            b1_array=np.array([-0.3, -0.3]),
            a2=-1.5,
            b2_array=np.array([0.3, 0.3]),
        )

    def test_mismatched_node_length(self):
        with pytest.raises(ValueError, match="node_array length"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0]),
                sigma_array=np.array([1.0, 1.0]),
                a1=1.5,
                b1_array=np.array([-0.3, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_non_increasing_nodes(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0, 0.0]),
                sigma_array=np.array([1.0, 1.0]),
                a1=1.5,
                b1_array=np.array([-0.3, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_node_not_starting_at_zero(self):
        with pytest.raises(ValueError, match="node_array\\[0\\] must be 0"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.1, 0.5]),
                sigma_array=np.array([1.0, 1.0]),
                a1=1.5,
                b1_array=np.array([-0.3, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_negative_sigma_array_rejected(self):
        with pytest.raises(ValueError, match="sigma_array must be non-negative"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0, 0.5]),
                sigma_array=np.array([1.0, -1.0]),
                a1=1.5,
                b1_array=np.array([-0.3, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_nan_boundary_array_rejected(self):
        with pytest.raises(ValueError, match="b1_array must contain only finite values"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0, 0.5]),
                sigma_array=np.array([1.0, 1.0]),
                a1=1.5,
                b1_array=np.array([np.nan, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_infinite_boundaries_rejected_by_default(self):
        with pytest.raises(ValueError, match="boundary intercepts must be finite"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0, 0.5]),
                sigma_array=np.array([1.0, 1.0]),
                a1=np.inf,
                b1_array=np.array([-0.3, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_multistage_model_accepts_infinite_boundaries(self):
        model = MultiStageModel(
            mu_array=np.array([1.0, 2.0]),
            node_array=np.array([0.0, 0.5]),
            sigma_array=np.array([1.0, 1.0]),
            a1=np.inf,
            b1_array=np.array([-0.3, -0.3]),
            a2=-np.inf,
            b2_array=np.array([0.3, 0.3]),
            x0=0.0,
        )
        assert np.isposinf(model.a1)
        assert np.isneginf(model.a2)

    def test_multistage_model_rejects_x0_outside_finite_boundaries(self):
        with pytest.raises(ValueError, match="x0 must be within"):
            MultiStageModel(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0, 0.5]),
                sigma_array=np.array([1.0, 1.0]),
                a1=1.5,
                b1_array=np.array([-0.3, -0.3]),
                a2=-1.5,
                b2_array=np.array([0.3, 0.3]),
                x0=2.0,
            )

    def test_invalid_finite_boundary_ordering_rejected(self):
        with pytest.raises(ValueError, match="initial upper boundary must be greater"):
            check_multistage_params(
                mu_array=np.array([1.0, 2.0]),
                node_array=np.array([0.0, 0.5]),
                sigma_array=np.array([1.0, 1.0]),
                a1=-1.0,
                b1_array=np.array([-0.3, -0.3]),
                a2=1.0,
                b2_array=np.array([0.3, 0.3]),
            )

    def test_accepts_single_stage_schedule(self):
        """check_multistage_params should accept a single-stage schedule."""
        check_multistage_params(
            np.array([0.2]),
            np.array([0.0]),
            np.array([1.0]),
            1.0,
            np.array([-0.2]),
            -1.0,
            np.array([0.2]),
        )
