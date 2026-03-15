"""High-level aDDM simulation API backed by a Cython inner loop.

Provides :func:`simulate_addm` for generating synthetic aDDM datasets and
the :class:`aDDModel` convenience wrapper around :class:`MultiStageModel`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import DDModel, MultiStageModel, piecewise_const_func
from .utils import get_alternating_mu_array
from .addm_simulator_cy import simulate_addm_batch_cy

_DEFAULT_CHUNK_SIZE = 200


class aDDModel(DDModel):
    """Attentional Drift Diffusion Model for a single trial.

    Wraps a piecewise-constant drift (alternating mu1/mu2 by fixation) with
    symmetric linear collapsing boundaries.  Inherits ``simulate_fpt_datum``
    and ``simulate_trajs`` from :class:`DDModel` for backward compatibility.
    """

    def __init__(self, mu1, mu2, sacc_array, flag, sigma, a, b, x0):
        super().__init__(x0)
        self.mu1 = mu1
        self.mu2 = mu2
        self.sacc_array = np.asarray(sacc_array, dtype=np.float64)
        self.flag = int(flag)
        self.d = len(self.sacc_array)
        self.mu_array = get_alternating_mu_array(mu1, mu2, self.d, self.flag)
        self.sigma = sigma
        self.a = a
        self.b = b

    def drift_coeff(self, X, t):
        return piecewise_const_func(t, self.mu_array, self.sacc_array)

    def diffusion_coeff(self, X, t):
        return self.sigma

    @property
    def is_update_vectorizable(self):
        return True

    def upper_bdy(self, t):
        return self.a - self.b * t

    def lower_bdy(self, t):
        return -self.a + self.b * t


def _build_mu_data_padded(
    mu1_data: NDArray,
    mu2_data: NDArray,
    d_data: NDArray,
    flag_data: NDArray,
    max_d: int,
) -> NDArray:
    """Vectorized construction of the (n_trials, max_d) drift-rate array."""
    n_trials = len(mu1_data)
    stages = np.arange(max_d)
    parity = (stages[np.newaxis, :] + flag_data[:, np.newaxis]) % 2
    mu_data = np.where(parity == 0, mu1_data[:, np.newaxis], mu2_data[:, np.newaxis])
    mask = stages[np.newaxis, :] < d_data[:, np.newaxis]
    mu_data = mu_data * mask
    return np.ascontiguousarray(mu_data, dtype=np.float64)


def _generate_fixation_sequences(
    rng: np.random.Generator,
    n_trials: int,
    max_t: float,
    gamma_shape: float,
    gamma_scale: float,
) -> tuple[NDArray, NDArray, int]:
    """Generate padded saccade-time arrays and per-trial stage counts.

    Returns (sacc_data_padded, d_data, max_d).
    """
    avg_fixation = gamma_shape * gamma_scale
    n_fixations_budget = int(max_t / avg_fixation) + 50
    n_fixations_budget = max(n_fixations_budget, 10)

    durations = rng.gamma(gamma_shape, gamma_scale, (n_trials, n_fixations_budget))
    cum_times = np.cumsum(durations, axis=1)
    sacc_times = np.concatenate(
        [np.zeros((n_trials, 1), dtype=np.float64), cum_times], axis=1
    )

    within_budget = sacc_times < max_t
    d_data = within_budget.sum(axis=1).astype(np.int32)

    if np.any(d_data < 1):
        raise ValueError(
            "Some trials have 0 fixations before max_t. "
            "Increase max_t or check gamma_shape/gamma_scale."
        )

    max_d = int(d_data.max())
    sacc_data_padded = np.ascontiguousarray(
        sacc_times[:, :max_d], dtype=np.float64
    )
    return sacc_data_padded, d_data, max_d


def simulate_addm(
    n_trials: int,
    eta: float,
    kappa: float,
    sigma: float,
    a: float,
    b: float,
    x0: float = 0.0,
    gamma_shape: float = 1.0,
    gamma_scale: float = 0.3,
    r_range: tuple[int, int] = (1, 6),
    dt: float = 1e-4,
    max_t: float = 20.0,
    n_threads: int = 1,
    random_state: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> dict:
    """Simulate *n_trials* of the attentional drift diffusion model.

    Uses a fast Cython inner loop for the Euler-Maruyama random walk and
    NumPy's Ziggurat sampler for all random draws.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    eta : float
        Attentional discount factor.
    kappa : float
        Drift-rate scaling.
    sigma : float
        Diffusion coefficient (noise).
    a : float
        Boundary intercept (half-width at t=0).
    b : float
        Boundary collapse slope (>=0).
    x0 : float
        Starting point of the evidence accumulator.
    gamma_shape, gamma_scale : float
        Parameters of the Gamma distribution for fixation durations.
    r_range : (int, int)
        Inclusive range for stimulus ratings (drawn uniformly).
    dt : float
        Euler-Maruyama time step.
    max_t : float
        Maximum trial duration (seconds).
    n_threads : int
        Must be 1 for now.  >1 raises ``NotImplementedError``.
    random_state : int or None
        Seed for reproducibility.
    chunk_size : int
        Trials processed per Cython call (controls peak memory).

    Returns
    -------
    dict with keys:
        rt, choice, mu_data_padded, sacc_data_padded,
        d_data, r1, r2, flag, mu1, mu2, params.
    """
    if n_threads > 1:
        raise NotImplementedError(
            "Multi-threaded simulation is not yet supported. Use n_threads=1."
        )

    rng = np.random.default_rng(random_state)

    # --- Stimulus values ---
    r1_data = rng.integers(r_range[0], r_range[1] + 1, size=n_trials)
    r2_data = rng.integers(r_range[0], r_range[1] + 1, size=n_trials)

    # --- Drift rates per trial ---
    mu1_data = kappa * (r1_data - eta * r2_data)
    mu2_data = kappa * (eta * r1_data - r2_data)

    # --- Fixation flag (which item first) ---
    flag_data = rng.integers(0, 2, size=n_trials).astype(np.int32)

    # --- Fixation sequences ---
    sacc_data_padded, d_data, max_d = _generate_fixation_sequences(
        rng, n_trials, max_t, gamma_shape, gamma_scale
    )

    # --- Drift-rate array ---
    mu_data_padded = _build_mu_data_padded(
        mu1_data.astype(np.float64),
        mu2_data.astype(np.float64),
        d_data,
        flag_data,
        max_d,
    )

    # --- Gaussian budget based on boundary geometry ---
    # With collapsing boundaries (b > 0), all trials must terminate before
    # the boundaries cross at t = a/b.  We can allocate only that many steps
    # instead of max_t/dt, which can be a large saving when a/b < max_t.
    # For flat boundaries (b == 0), we must allocate for the full max_t.
    if b > 0:
        budget_time = min(a / b, max_t)
    else:
        budget_time = max_t
    budget_steps = int(budget_time / dt) + 1

    rt_all = np.empty(n_trials, dtype=np.float64)
    choice_all = np.empty(n_trials, dtype=np.int32)

    for start in range(0, n_trials, chunk_size):
        end = min(start + chunk_size, n_trials)
        n_chunk = end - start

        gaussian_chunk = np.ascontiguousarray(
            rng.standard_normal((n_chunk, budget_steps), dtype=np.float64)
        )

        rt_chunk, choice_chunk = simulate_addm_batch_cy(
            np.ascontiguousarray(mu_data_padded[start:end]),
            np.ascontiguousarray(sacc_data_padded[start:end]),
            np.ascontiguousarray(d_data[start:end]),
            sigma, a, b, x0, dt, budget_time,
            gaussian_chunk,
        )

        rt_all[start:end] = rt_chunk
        choice_all[start:end] = choice_chunk

    # --- Post-process: truncate sacc_data to actual RTs (vectorized) ---
    terminated = rt_all > 0
    rt_col = rt_all[:, np.newaxis]
    beyond_mask = (sacc_data_padded >= rt_col) & terminated[:, np.newaxis]
    sacc_data_padded[beyond_mask] = 0.0

    stage_indices = np.arange(max_d)[np.newaxis, :]
    active_before_rt = (stage_indices < d_data[:, np.newaxis]) & (
        sacc_data_padded < rt_col
    )
    d_new = active_before_rt.sum(axis=1).astype(np.int32)
    d_data = np.where(terminated, np.maximum(d_new, 1), d_data).astype(np.int32)

    mu_data_padded = _build_mu_data_padded(
        mu1_data.astype(np.float64),
        mu2_data.astype(np.float64),
        d_data,
        flag_data,
        max_d,
    )

    return {
        "rt": rt_all,
        "choice": choice_all,
        "mu_data_padded": mu_data_padded,
        "sacc_data_padded": sacc_data_padded,
        "d_data": d_data,
        "r1": r1_data,
        "r2": r2_data,
        "flag": flag_data,
        "mu1": mu1_data,
        "mu2": mu2_data,
        "params": {
            "eta": eta,
            "kappa": kappa,
            "sigma": sigma,
            "a": a,
            "b": b,
            "x0": x0,
            "dt": dt,
            "max_t": max_t,
            "gamma_shape": gamma_shape,
            "gamma_scale": gamma_scale,
            "r_range": r_range,
        },
    }
