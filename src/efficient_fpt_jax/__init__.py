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
from .multi_stage import get_addm_fptd_jax, pad_sacc_array_safely
from .batch import compute_likelihoods_batch, compute_nll_batch
from .utils import lgwt_lookup_table, GAUSS_LEGENDRE_30_X, GAUSS_LEGENDRE_30_W

__all__ = [
    # Single-stage functions
    "fptd_basic_jax",
    "q_basic_jax", 
    "fptd_single_jax",
    "q_single_jax",
    # Multi-stage functions
    "get_addm_fptd_jax",
    "pad_sacc_array_safely",  # Helper for understanding safe padding
    # Batch computation
    "compute_likelihoods_batch",
    "compute_nll_batch",
    # Utilities
    "lgwt_lookup_table",
    "GAUSS_LEGENDRE_30_X",
    "GAUSS_LEGENDRE_30_W",
]

