from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def check_valid_multistage_params(mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array):
    """
    Check the validity of the parameters for the multi-stage model.
    """
    d = len(mu_array)
    if len(sacc_array) != d:
        raise ValueError(f"sacc_array length {len(sacc_array)} != mu_array length {d}")
    if len(sigma_array) != d:
        raise ValueError(f"sigma_array length {len(sigma_array)} != mu_array length {d}")
    if len(b1_array) != d:
        raise ValueError(f"b1_array length {len(b1_array)} != mu_array length {d}")
    if len(b2_array) != d:
        raise ValueError(f"b2_array length {len(b2_array)} != mu_array length {d}")
    if not all(np.diff(sacc_array) > 0):
        raise ValueError("sacc_array must be strictly increasing")
    if sacc_array[0] != 0:
        raise ValueError("sacc_array[0] must be 0")
    if d >= 2 and sacc_array[1] <= 0:
        raise ValueError("sacc_array[1] must be positive")


def get_alternating_mu_array(mu1, mu2, d):
    """
    Generate a list of alternating drift rates with length `d`.
    Always starts with mu1: mu1 -> mu2 -> mu1 -> ...
    If the opposite ordering is needed, swap mu1 and mu2 at the call site.
    """
    stages = np.arange(d)
    return np.where(stages % 2 == 0, mu1, mu2)


def build_mu_array_data(mu1_data, mu2_data, d_data, max_d):
    """Vectorized construction of the (n_trials, max_d) drift-rate array.

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


def adaptive_interpolation(f, x_range, error_threshold, max_iterations=1000, initial_points=10, num_eval_points=1000):
    """
    Adaptive linear interpolation of a function `f` over a specified range `x_range`.
    The function iteratively refines the interpolation points until the maximum error
    on the evaluation grid is below the specified `error_threshold` or the maximum number of iterations is reached.
    The function returns the x-coordinates and corresponding y-coordinates of the interpolation.
    """
    x_points = np.linspace(x_range[0], x_range[1], initial_points)
    y_points = f(x_points)
    xi = np.linspace(x_range[0], x_range[1], num_eval_points)
    yi = np.interp(xi, x_points, y_points)
    iteration = 0
    while iteration < max_iterations:
        f_actual = f(xi)
        errors = np.abs(yi - f_actual)
        max_error = np.max(errors)
        if max_error <= error_threshold:
            break
        max_error_idx = np.argmax(errors)
        new_x = xi[max_error_idx]
        if np.any(np.isclose(x_points, new_x, atol=1e-12)):
            break
        new_y = f(new_x)
        idx = np.searchsorted(x_points, new_x)
        x_points = np.insert(x_points, idx, new_x)
        y_points = np.insert(y_points, idx, new_y)
        yi = np.interp(xi, x_points, y_points)
        iteration += 1
    else:
        print("Warning: Maximum iterations reached before meeting error threshold.")
    return x_points, y_points


# ---------------------------------------------------------------------------
# aDDM experiment generation
# ---------------------------------------------------------------------------

def _generate_fixation_sequences(
    rng: np.random.Generator,
    n_trials: int,
    T: float,
    gamma_shape: float,
    gamma_scale: float,
) -> tuple[NDArray, NDArray, int]:
    """Generate padded saccade-time arrays and per-trial stage counts.

    Returns (sacc_array_data, d_data, max_d).
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
    sacc_array_data = np.ascontiguousarray(
        sacc_times[:, :max_d], dtype=np.float64
    )
    return sacc_array_data, d_data, max_d


def generate_addm_experiment(
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
    T: float = 20.0,
    n_threads: int = 1,
    random_state: int | None = None,
    chunk_size: int | None = None,
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
    T : float
        Maximum trial duration (seconds).
    n_threads : int
        Number of OpenMP threads for the Cython simulator (1 = serial).
    random_state : int or None
        Seed for reproducibility.
    chunk_size : int
        Trials processed per Cython call (controls peak memory).

    Returns
    -------
    dict with keys:
        rt, choice, mu_array_data, sacc_array_data,
        d_data, r1, r2, flag, mu1, mu2, params.
    """
    # Lazy import to avoid circular dependency (models imports utils)
    from .models import aDDModel, _DEFAULT_CHUNK_SIZE, _swap_and_build_mu

    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

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
    sacc_array_data, d_data, max_d = _generate_fixation_sequences(
        rng, n_trials, T, gamma_shape, gamma_scale
    )

    # --- Simulate via aDDModel ---
    model = aDDModel(eta, kappa, sigma, a, b, x0)
    rt_all, choice_all, _ = model.simulate_fpt(
        r1_data, r2_data, flag_data, sacc_array_data, d_data,
        T=T, dt=dt, rng=rng, chunk_size=chunk_size, n_threads=n_threads,
    )

    # --- Post-process: truncate to stages that started before RT ---
    terminated = rt_all > 0
    rt_col = rt_all[:, np.newaxis]
    stage_indices = np.arange(max_d)[np.newaxis, :]

    # Compute new stage counts using ORIGINAL (uncorrupted) sacc times
    active_before_rt = (stage_indices < d_data[:, np.newaxis]) & (
        sacc_array_data < rt_col
    )
    d_new = active_before_rt.sum(axis=1).astype(np.int32)
    d_data = np.where(terminated, np.maximum(d_new, 1), d_data).astype(np.int32)

    # Now zero out entries beyond the updated d_data
    beyond_new_d = stage_indices >= d_data[:, np.newaxis]
    sacc_array_data[beyond_new_d] = 0.0

    # Recompute max_d and trim arrays to remove fully-zero trailing columns
    max_d = int(d_data.max())
    sacc_array_data = np.ascontiguousarray(sacc_array_data[:, :max_d])

    mu_array_data = _swap_and_build_mu(mu1_data, mu2_data, flag_data, d_data, max_d)

    return {
        "rt": rt_all,
        "choice": choice_all,
        "mu_array_data": mu_array_data,
        "sacc_array_data": sacc_array_data,
        "d_data": d_data,
        "r1_data": r1_data,
        "r2_data": r2_data,
        "flag_data": flag_data,
        "mu1_data": mu1_data,
        "mu2_data": mu2_data,
        "params": {
            "eta": eta,
            "kappa": kappa,
            "sigma": sigma,
            "a": a,
            "b": b,
            "x0": x0,
            "dt": dt,
            "T": T,
            "gamma_shape": gamma_shape,
            "gamma_scale": gamma_scale,
            "r_range": r_range,
        },
    }
