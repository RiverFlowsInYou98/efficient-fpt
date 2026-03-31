"""ADDM-specific helper functions for drift-rate construction and experiment generation.

Centralizes all NumPy-based ADDM drift-rate array construction (single-trial
and batched) as well as the experiment simulator that wraps ``aDDModel``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Drift-rate construction
# ---------------------------------------------------------------------------


def _build_alternating_mu_array(mu1, mu2, d):
    """Generate a 1-D array of alternating drift rates with length *d*.

    Always starts with mu1: ``[mu1, mu2, mu1, ...]``.
    If the opposite ordering is needed, swap mu1 and mu2 at the call site.
    """
    stages = np.arange(d)
    return np.where(stages % 2 == 0, mu1, mu2)


def _build_alternating_mu_array_data(mu1_data, mu2_data, d_data, max_d):
    """Vectorized construction of the (n_trials, max_d) alternating drift-rate array.

    Always starts with mu1: ``[mu1, mu2, mu1, ...]``.  If the opposite
    ordering is needed for some trials, swap mu1/mu2 at the call site.
    """
    if max_d == 0:
        return np.empty((len(mu1_data), 0), dtype=np.float64)
    stages = np.arange(max_d)
    parity = stages[np.newaxis, :] % 2
    mu_array_data = np.where(
        parity == 0,
        np.asarray(mu1_data, dtype=np.float64)[:, np.newaxis],
        np.asarray(mu2_data, dtype=np.float64)[:, np.newaxis],
    )
    mask = stages[np.newaxis, :] < np.asarray(d_data)[:, np.newaxis]
    mu_array_data = mu_array_data * mask
    return np.ascontiguousarray(mu_array_data, dtype=np.float64)


def _build_addm_mu_array(eta, kappa, r1, r2, flag, d):
    """Build a single-trial ADDM drift-rate array from covariates.

    Derives mu1/mu2 from ADDM parameters, swaps based on *flag*,
    and returns an alternating array of length *d*.

    Parameters
    ----------
    eta : float
        Attentional discount factor.
    kappa : float
        Drift-rate scaling.
    r1, r2 : float
        Stimulus ratings for item 1 and item 2.
    flag : int
        0 = fixate item 1 first, 1 = fixate item 2 first.
    d : int
        Number of stages.
    """
    mu1 = kappa * (r1 - eta * r2)
    mu2 = kappa * (eta * r1 - r2)
    mu_first = mu1 if flag == 0 else mu2
    mu_second = mu2 if flag == 0 else mu1
    return _build_alternating_mu_array(mu_first, mu_second, d)


def _build_addm_mu_array_data(eta, kappa, r1_data, r2_data, flag_data, d_data, max_d):
    """Build the padded (n_trials, max_d) ADDM drift array from covariates.

    Derives mu1/mu2 from ADDM parameters, swaps based on *flag_data*,
    and builds the alternating drift-rate array for each trial.
    """
    r1_data = np.asarray(r1_data, dtype=np.float64)
    r2_data = np.asarray(r2_data, dtype=np.float64)
    mu1_data = kappa * (r1_data - eta * r2_data)
    mu2_data = kappa * (eta * r1_data - r2_data)
    mu1_eff = np.where(flag_data == 0, mu1_data, mu2_data).astype(np.float64)
    mu2_eff = np.where(flag_data == 0, mu2_data, mu1_data).astype(np.float64)
    return _build_alternating_mu_array_data(mu1_eff, mu2_eff, d_data, max_d)


# ---------------------------------------------------------------------------
# aDDM experiment generation
# ---------------------------------------------------------------------------


def _generate_sacc_array_data(
    rng: np.random.Generator,
    n_trials: int,
    T: float,
    gamma_shape: float,
    gamma_scale: float,
) -> tuple[NDArray, NDArray, int]:
    """Generate padded saccade-time arrays and per-trial stage counts.

    Returns (sacc_array_data, d_data, max_d).
    `sacc_array_data` is of shape(n_trials, max_d), of which only the first `d` columns are active.
    d=1 corresponds to no saccades by time `T`, and the active sacc_array would be just [0.0].
    """
    avg_fixation = gamma_shape * gamma_scale
    n_fixations_budget = int(T / avg_fixation) + 50
    n_fixations_budget = max(n_fixations_budget, 10)

    durations = rng.gamma(gamma_shape, gamma_scale, (n_trials, n_fixations_budget))
    cum_times = np.cumsum(durations, axis=1)
    sacc_times = np.concatenate(
        [np.zeros((n_trials, 1), dtype=np.float64), cum_times], axis=1
    )

    within_budget = sacc_times < T
    d_data = within_budget.sum(axis=1).astype(np.int32)

    if np.any(d_data < 1):
        raise ValueError(
            "Some trials have 0 fixations before T. "
            "Increase T or check gamma_shape/gamma_scale."
        )

    max_d = int(d_data.max())
    sacc_array_data = np.ascontiguousarray(sacc_times[:, :max_d], dtype=np.float64)
    return sacc_array_data, d_data, max_d


