"""Shared fixtures for efficient-fpt test suite."""

import pytest
import numpy as np


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


# ---------------------------------------------------------------------------
# Backend import fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def jax_backend():
    """Import guard + convenience accessor for the JAX backend."""
    pytest.importorskip("jax")
    import efficient_fpt.jax as jx
    return jx


@pytest.fixture
def cython_multi_stage():
    """Import guard for Cython multi_stage functions. Returns (compute_addm_fptd, compute_heterog_multistage_fptd) or skips."""
    try:
        from efficient_fpt.cython.multi_stage import (
            compute_addm_fptd,
            compute_heterog_multistage_fptd,
        )
        return compute_addm_fptd, compute_heterog_multistage_fptd
    except ImportError:
        pytest.skip("Cython backend not available")


@pytest.fixture
def cython_batch():
    """Import guard for Cython batch functions."""
    try:
        from efficient_fpt.cython import batch
        return batch
    except ImportError:
        pytest.skip("Cython backend not available")


# ---------------------------------------------------------------------------
# Common ADDM parameter sets
# ---------------------------------------------------------------------------

@pytest.fixture
def addm_params():
    """Standard ADDM model parameters used across many tests.

    Returns dict with keys: eta, kappa, sigma, a, b, x0.
    """
    return dict(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)


@pytest.fixture
def single_stage_params():
    """Standard single-stage model parameters.

    Returns dict with keys: mu, sigma, a, b, x0.
    """
    return dict(mu=1.0, sigma=1.0, a=1.5, b=0.3, x0=-0.5)
