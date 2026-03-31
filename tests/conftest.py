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
