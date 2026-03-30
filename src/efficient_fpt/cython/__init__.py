"""Cython-accelerated implementations of first-passage time density computation."""

from .single_stage import fptd_basic, q_basic, fptd_single, q_single
from .multi_stage import (
    compute_addm_fptd,
    compute_heterog_multistage_fptd,
)
from .utils import print_num_threads
from .batch import (
    compute_addm_likelihoods,
    compute_addm_nll,
    compute_addm_mean_nll,
    compute_addm_sum_nll,
    compute_tada_likelihoods,
    compute_tada_mean_nll,
)
from .simulator import (
    simulate_homog_ddm_fpt,
    simulate_heterog_multistage_fpt,
    _simulate_addm_fpt,
)

__all__ = [
    # Single-stage
    "fptd_basic",
    "q_basic",
    "fptd_single",
    "q_single",
    # Multi-stage
    "compute_addm_fptd",
    "compute_heterog_multistage_fptd",
    # Batch computation
    "compute_addm_likelihoods",
    "compute_addm_nll",
    "compute_addm_mean_nll",
    "compute_addm_sum_nll",
    "compute_tada_likelihoods",
    "compute_tada_mean_nll",
    # Simulation
    "simulate_homog_ddm_fpt",
    "simulate_heterog_multistage_fpt",
    "_simulate_addm_fpt",
    # Utilities
    "print_num_threads",
]
