"""Cython-accelerated implementations of first-passage time density computation."""

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
    compute_heterog_multistage_logfptd,
)
from .utils import get_num_threads, print_num_threads
from .batch import (
    compute_addm_loglikelihoods,
    compute_addm_nll,
    compute_addm_mean_nll,
    compute_addm_sum_nll,
    compute_tada_loglikelihoods,
    compute_tada_mean_nll,
)
from .simulator import (
    simulate_homog_ddm_fpt,
    simulate_heterog_multistage_fpt,
)

__all__ = [
    # Single-stage
    "fptd_basic",
    "q_basic",
    "fptd_single",
    "q_single",
    "log_fptd_basic",
    "log_q_basic",
    "log_fptd_single",
    "log_q_single",
    # Multi-stage
    "compute_addm_logfptd",
    "compute_heterog_multistage_logfptd",
    # Batch computation
    "compute_addm_loglikelihoods",
    "compute_addm_nll",
    "compute_addm_mean_nll",
    "compute_addm_sum_nll",
    "compute_tada_loglikelihoods",
    "compute_tada_mean_nll",
    # Simulation
    "simulate_homog_ddm_fpt",
    "simulate_heterog_multistage_fpt",
    # Utilities
    "get_num_threads",
    "print_num_threads",
]
