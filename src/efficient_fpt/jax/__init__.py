"""JAX implementation of efficient first-passage time density computation.

Provides JAX-compatible implementations enabling:
- GPU acceleration
- Automatic differentiation for gradient-based inference
- Integration with JAX-based probabilistic programming libraries

Importing this package does not mutate global JAX precision settings. Use
``set_jax_precision(...)`` explicitly when a caller wants to opt into x64.
"""

from .single_stage import (
    fptd_basic,
    q_basic,
    fptd_single,
    q_single,
    log_fptd_basic,
    log_q_basic,
    log_fptd_single,
    log_q_single,
)
from .multi_stage import (
    compute_addm_logfptd,
    compute_addm_logfptd_precomputed,
    compute_addm_logfptd_stagescan,
    compute_heterog_multistage_logfptd,
    compute_heterog_multistage_logfptd_precomputed,
    compute_heterog_multistage_logfptd_stagescan,
)
from .batch import (
    compute_addm_loglikelihoods_batchvmap,
    compute_addm_loglikelihoods_batchscan,
    compute_addm_loglikelihoods,
    make_addm_nll_function_batchvmap,
    make_addm_nll_function_batchscan,
    compute_addm_nll,
    make_addm_nll_function,
)
from .utils import (
    lgwt_lookup_table,
    get_gauss_legendre_ref,
    GAUSS_LEGENDRE_30_X,
    GAUSS_LEGENDRE_30_W,
    set_jax_precision,
    get_jax_dtype,
)

__all__ = [
    # Single-stage functions
    "fptd_basic",
    "q_basic",
    "fptd_single",
    "q_single",
    "log_fptd_basic",
    "log_q_basic",
    "log_fptd_single",
    "log_q_single",
    # Multi-stage functions
    "compute_addm_logfptd",
    "compute_addm_logfptd_precomputed",
    "compute_addm_logfptd_stagescan",
    "compute_heterog_multistage_logfptd",
    "compute_heterog_multistage_logfptd_precomputed",
    "compute_heterog_multistage_logfptd_stagescan",
    # Batch computation
    "compute_addm_loglikelihoods_batchvmap",
    "compute_addm_loglikelihoods_batchscan",
    "compute_addm_loglikelihoods",
    "make_addm_nll_function_batchvmap",
    "make_addm_nll_function_batchscan",
    "compute_addm_nll",
    "make_addm_nll_function",
    # Utilities
    "lgwt_lookup_table",
    "get_gauss_legendre_ref",
    "GAUSS_LEGENDRE_30_X",
    "GAUSS_LEGENDRE_30_W",
    # Precision control
    "set_jax_precision",
    "get_jax_dtype",
]
