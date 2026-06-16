"""Shared fixtures for efficient-fpt test suite."""

import pytest


# ---------------------------------------------------------------------------
# Global test configuration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def configure_jax_precision_for_tests():
    """Opt test runs into JAX x64 explicitly when JAX is available.

    The package itself no longer mutates global JAX precision at import time.
    The numerical agreement tests, however, were written assuming float64
    precision, so the test suite enables x64 deliberately via the public
    precision helper.
    """
    try:
        import jax
        from efficient_fpt.jax.utils import set_jax_precision
    except ImportError:
        yield
        return

    previous = bool(jax.config.read("jax_enable_x64"))
    set_jax_precision(True)
    yield
    set_jax_precision(previous)


@pytest.fixture
def addm_model():
    """Standard aDDModel for testing."""
    from efficient_fpt.models import aDDModel
    return aDDModel(eta=0.5, kappa=1.0, sigma=1.0, a=1.0, b=0.5, x0=0.0)


@pytest.fixture
def addm_experiment(addm_model):
    """Generate a small aDDM experiment for testing."""
    data = addm_model.generate_experiment(n_trials=50, rng=42)
    # Cast r1/r2 to float64 for Cython compatibility
    data["covariates"]["r1_data"] = data["covariates"]["r1_data"].astype("float64")
    data["covariates"]["r2_data"] = data["covariates"]["r2_data"].astype("float64")
    return data
