"""Utility functions and constants for JAX implementation of efficient-fpt.

Wraps shared Gauss-Legendre quadrature constants as JAX arrays.
Provides a precision switch (float64 by default, configurable to float32).
"""

import numpy as np
import jax
import jax.numpy as jnp
from ..quadrature import (
    GAUSS_LEGENDRE_30_X as _NP_GL30_X,
    GAUSS_LEGENDRE_30_W as _NP_GL30_W,
    lgwt_lookup_table as _np_lgwt,
)

# ---------------------------------------------------------------------------
# Precision control
# ---------------------------------------------------------------------------


def _current_jax_dtype():
    """Return the active JAX floating-point dtype without mutating config."""
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


_dtype = _current_jax_dtype()

# ---------------------------------------------------------------------------
# Numerical safety constants for stage-duration computations
# ---------------------------------------------------------------------------

_DUMMY_STAGE_DURATION = 1.0  # placeholder duration for invalid/padding stages


def _refresh_quadrature_constants():
    """Refresh cached quadrature constants to match the current dtype."""
    global GAUSS_LEGENDRE_30_X, GAUSS_LEGENDRE_30_W
    GAUSS_LEGENDRE_30_X = jnp.array(_NP_GL30_X, dtype=_dtype)
    GAUSS_LEGENDRE_30_W = jnp.array(_NP_GL30_W, dtype=_dtype)


def set_jax_precision(use_x64: bool = True):
    """Set the floating-point precision for all JAX FPT computations.

    Parameters
    ----------
    use_x64 : bool, optional (default=True)
        If True, use float64 (recommended for accuracy).
        If False, use float32 (faster on GPU, lower precision).

    Notes
    -----
    This helper updates the process-wide JAX x64 flag and refreshes the cached
    quadrature constants used by this package. Call it explicitly before
    compiling or benchmarking JAX workloads when a specific precision mode is
    required.
    set_jax_precision(True) enables FP64 for better numerical accuracy, but may
    be much slower on consumer/workstation GPUs with limited FP64 throughput and
    is mainly recommended for CPU or datacenter/HPC GPUs.
    """
    global _dtype, _QUAD_CACHE
    jax.config.update("jax_enable_x64", use_x64)
    _dtype = _current_jax_dtype()
    _QUAD_CACHE.clear()
    _refresh_quadrature_constants()


def get_jax_dtype():
    """Return the current floating-point dtype used by JAX FPT functions."""
    return _dtype


def positive_log(values):
    """Return log(x) for x > 0, -inf for x <= 0, and nan when x is nan."""
    values = jnp.asarray(values, dtype=_dtype)
    safe = jnp.where(values > 0.0, values, 1.0)
    logs = jnp.where(values > 0.0, jnp.log(safe), -jnp.inf)
    return jnp.where(jnp.isnan(values), jnp.nan, logs)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------

# These module-level constants are JAX arrays, which are immutable by design
# (item assignment raises TypeError), so no additional write-protection is needed.
GAUSS_LEGENDRE_30_X = None
GAUSS_LEGENDRE_30_W = None
_refresh_quadrature_constants()

_QUAD_CACHE = {}


def get_gauss_legendre_ref(order: int):
    """Return (x_ref, w_ref) on [-1, 1] as JAX arrays for the given order.

    The shared cache stores NumPy reference arrays, not JAX arrays. This is
    important because ``get_gauss_legendre_ref(...)`` is called from jitted /
    transformed code paths. Storing ``jnp.array(...)`` results in global state
    during tracing can leak tracers and break later JAX transforms.

    Precision changes are handled at the conversion boundary by casting the
    cached NumPy arrays to the current JAX dtype on return.

    Parameters
    ----------
    order : int
        Quadrature order (any order supported by the shared cache)

    Returns
    -------
    x_ref : jnp.ndarray of shape (order,)
        Reference nodes on [-1, 1]
    w_ref : jnp.ndarray of shape (order,)
        Reference weights on [-1, 1]
    """
    if order not in _QUAD_CACHE:
        x_np, w_np = _np_lgwt(order, -1.0, 1.0)
        _QUAD_CACHE[order] = (
            np.asarray(x_np, dtype=np.float64),
            np.asarray(w_np, dtype=np.float64),
        )
    x_np, w_np = _QUAD_CACHE[order]
    return jnp.asarray(x_np, dtype=_dtype), jnp.asarray(w_np, dtype=_dtype)


def lgwt_lookup_table(order: int, a: float, b: float):
    """Return scaled Gauss-Legendre nodes and weights for interval [a, b] as JAX arrays.

    Parameters
    ----------
    order : int
        Order of quadrature (any order supported by the shared cache)
    a : float
        Lower bound of integration interval
    b : float
        Upper bound of integration interval

    Returns
    -------
    x : jnp.ndarray
        Quadrature nodes scaled to [a, b]
    w : jnp.ndarray
        Quadrature weights scaled for [a, b]
    """
    x_np, w_np = _np_lgwt(order, a, b)
    return jnp.array(x_np, dtype=_dtype), jnp.array(w_np, dtype=_dtype)


def get_jax_device_name():
    """Return a human-readable name for the default JAX device.

    For GPU devices, queries ``nvidia-smi`` for the actual model name.
    Falls back to the JAX device repr if ``nvidia-smi`` is unavailable.
    """
    import jax

    dev = jax.devices()[0]
    if dev.platform == "gpu":
        try:
            import subprocess

            name = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    text=True,
                )
                .strip()
                .split("\n")[0]
            )
            return f"{name} (GPU)"
        except Exception:
            return f"{dev!r} (GPU)"
    return repr(dev)
