"""
JAX implementation of efficient first-passage time density computation.

This package provides JAX-compatible implementations of the algorithms
from the efficient-fpt package, enabling:
- GPU acceleration
- Automatic differentiation for gradient-based inference
- Integration with JAX-based probabilistic programming libraries
"""

from .single_stage import (
    fptd_basic_jax,
    q_basic_jax,
    fptd_single_jax,
    q_single_jax,
)
from .multi_stage import get_addm_fptd_jax, get_addm_fptd_jax_fast, pad_sacc_array_safely
from .utils import GAUSS_LEGENDRE_30_X, GAUSS_LEGENDRE_30_W

__all__ = [
    # Single-stage functions
    "fptd_basic_jax",
    "q_basic_jax", 
    "fptd_single_jax",
    "q_single_jax",
    # Multi-stage functions
    "get_addm_fptd_jax",
    "get_addm_fptd_jax_fast",  # Optimized version with faster gradients
    "pad_sacc_array_safely",
    # Utilities
    "GAUSS_LEGENDRE_30_X",
    "GAUSS_LEGENDRE_30_W",
]

