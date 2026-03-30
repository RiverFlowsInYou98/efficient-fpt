# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Euler-Maruyama first-passage-time simulators with inline RNG and OpenMP.

This module provides three public entry points, each targeting a different
level of model generality:

* **simulate_homog_ddm_fpt** — *Homogeneous* DDM.  All trials share the
  same precomputed drift, diffusion, and boundary arrays (evaluated on a
  common time grid).  Only the starting position and PRNG seed vary across
  trials.  Used by ``DDModel.simulate_fpt`` for single-stage models.

* **simulate_heterog_multistage_fpt** — *General heterogeneous multi-stage*
  DDM.  Each trial has its own per-stage drift, diffusion, boundary
  intercepts, and boundary slopes, stored in padded 2-D arrays.  Stage
  onset times (``node_array_data``) mark when parameters switch.  This is
  the most flexible entry point and is called by ``_simulate_addm_fpt``.

* **_simulate_addm_fpt** — *Attentional DDM* (aDDM) convenience wrapper.
  Accepts the compact aDDM parameterisation (scalar ``sigma``, ``a``,
  ``b``) and saccade onset times (``sacc_array_data``), expands them into
  the full per-stage arrays, and delegates to
  ``simulate_heterog_multistage_fpt``.  Used by
  ``aDDModel.simulate_fpt``.

All three use the same simulation engine:

1. **Inline C-level PRNG** — xoshiro256++ seeded per trial via SplitMix64,
   with Box-Muller transform for Gaussian draws.  No pre-allocated noise
   matrix is needed, keeping memory usage O(n_trials) instead of
   O(n_trials × max_steps).

2. **OpenMP parallelism** — Trials are distributed across threads via
   ``prange`` with dynamic scheduling.  Each thread has its own RNG state,
   so results are deterministic for a given seed array regardless of
   ``n_threads``.

The per-trial logic lives in two ``cdef nogil`` helpers that are called
from inside the ``prange`` loop and never touch the GIL:

* ``_run_homog_trial`` — **step-indexed**.  Drift, diffusion, and
  boundaries are pre-evaluated on a shared time grid (one value per
  discrete time step).  Supports arbitrary time-dependent parameters
  (e.g., Weibull survival boundaries) by table lookup at each step.

* ``_run_heterog_trial`` — **stage-indexed**.  Drift and diffusion are
  piecewise-constant per stage; boundaries are piecewise-linear
  (intercept + slope × time-since-stage-onset).  Stage transitions
  occur when the particle clock passes the next ``node_array`` entry.
  Much more memory-efficient for multi-stage models (``max_d`` ≪
  ``max_steps``) but limited to piecewise-constant/linear parameters.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, M_PI
from libc.stdint cimport uint64_t
from cython.parallel cimport prange


# ---------------------------------------------------------------------------
# C-level PRNG: xoshiro256++ with SplitMix64 seeding + Box-Muller transform
# ---------------------------------------------------------------------------

cdef struct Xoshiro256State:
    uint64_t s0
    uint64_t s1
    uint64_t s2
    uint64_t s3


cdef inline uint64_t _rotl(uint64_t x, int k) noexcept nogil:
    return (x << k) | (x >> (64 - k))


cdef inline uint64_t xoshiro256pp_next(Xoshiro256State *state) noexcept nogil:
    cdef uint64_t result = _rotl(state.s0 + state.s3, 23) + state.s0
    cdef uint64_t t = state.s1 << 17
    state.s2 ^= state.s0
    state.s3 ^= state.s1
    state.s1 ^= state.s2
    state.s0 ^= state.s3
    state.s2 ^= t
    state.s3 = _rotl(state.s3, 45)
    return result


cdef inline uint64_t splitmix64_next(uint64_t *state) noexcept nogil:
    state[0] += <uint64_t>0x9e3779b97f4a7c15
    cdef uint64_t z = state[0]
    z = (z ^ (z >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * <uint64_t>0x94d049bb133111eb
    return z ^ (z >> 31)


cdef inline void seed_xoshiro256(Xoshiro256State *state, uint64_t seed) noexcept nogil:
    cdef uint64_t sm_state = seed
    state.s0 = splitmix64_next(&sm_state)
    state.s1 = splitmix64_next(&sm_state)
    state.s2 = splitmix64_next(&sm_state)
    state.s3 = splitmix64_next(&sm_state)


cdef inline double uint64_to_double(uint64_t x) noexcept nogil:
    return <double>(x >> 11) * (1.0 / 9007199254740992.0)  # 2^53


cdef struct BoxMullerState:
    double spare
    int has_spare


cdef inline double box_muller_next(Xoshiro256State *rng_state, BoxMullerState *bm_state) noexcept nogil:
    cdef double u1, u2, mag
    if bm_state.has_spare:
        bm_state.has_spare = 0
        return bm_state.spare
    u1 = uint64_to_double(xoshiro256pp_next(rng_state))
    u2 = uint64_to_double(xoshiro256pp_next(rng_state))
    # Guard against log(0)
    if u1 < 1e-300:
        u1 = 1e-300
    mag = sqrt(-2.0 * log(u1))
    bm_state.spare = mag * sin(2.0 * M_PI * u2)
    bm_state.has_spare = 1
    return mag * cos(2.0 * M_PI * u2)





# ---------------------------------------------------------------------------
# Per-trial helper functions (called from OpenMP prange)
# ---------------------------------------------------------------------------

cdef void _run_homog_trial(
    double[::1] drift_vals,
    double[::1] diffusion_vals,
    double[::1] upper_vals,
    double[::1] lower_vals,
    double x0,
    double dt,
    int max_steps,
    double T,
    uint64_t seed,
    double *rt_out,
    int *choice_out,
    double *x_final_out,
) noexcept nogil:
    """Run a single trial using step-indexed parameter arrays (nogil).

    Parameters are pre-evaluated on a shared time grid — one value per
    discrete step — so drift, diffusion, and boundaries can be *arbitrary*
    functions of time (looked up by step index).  Compare with
    ``_run_heterog_trial`` which is stage-indexed and assumes
    piecewise-constant drift and piecewise-linear boundaries.
    """
    cdef:
        Xoshiro256State rng_state
        BoxMullerState bm_state
        double y, z, t_particle, dt_curr, sqrt_dt_curr, half_dt_curr
        int step

    seed_xoshiro256(&rng_state, seed)
    bm_state.has_spare = 0
    y = x0
    t_particle = 0.0
    rt_out[0] = -1.0
    choice_out[0] = 0

    for step in range(max_steps):
        dt_curr = T - t_particle
        if dt_curr <= 0.0:
            break
        if dt_curr > dt:
            dt_curr = dt
        sqrt_dt_curr = sqrt(dt_curr)
        half_dt_curr = 0.5 * dt_curr
        z = box_muller_next(&rng_state, &bm_state)
        y = y + drift_vals[step] * dt_curr + diffusion_vals[step] * sqrt_dt_curr * z
        t_particle = t_particle + dt_curr

        if y >= upper_vals[step]:
            rt_out[0] = t_particle - half_dt_curr
            choice_out[0] = 1
            break
        elif y <= lower_vals[step]:
            rt_out[0] = t_particle - half_dt_curr
            choice_out[0] = -1
            break

    x_final_out[0] = y


cdef void _run_heterog_trial(
    double[:, ::1] mu_array_data,
    double[:, ::1] sigma_array_data,
    double[:, ::1] node_array_data,
    int d,
    double[:, ::1] ub_array_data,
    double[:, ::1] b1_array_data,
    double[:, ::1] lb_array_data,
    double[:, ::1] b2_array_data,
    int trial_idx,
    double x0,
    double dt,
    int max_steps,
    double T,
    uint64_t seed,
    double *rt_out,
    int *choice_out,
    double *x_final_out,
) noexcept nogil:
    """Run a single trial using stage-indexed parameter arrays (nogil).

    Drift and diffusion are piecewise-constant per stage; boundaries are
    piecewise-linear (intercept + slope × time-since-stage-onset).  Stage
    transitions occur when elapsed time passes the next ``node_array``
    entry.  Much more memory-efficient than ``_run_homog_trial`` for
    multi-stage models (``max_d`` ≪ ``max_steps``), but limited to
    piecewise-constant/linear parameters.
    """
    cdef:
        Xoshiro256State rng_state
        BoxMullerState bm_state
        double y, z, t_particle, upper, lower, dt_curr, sqrt_dt_curr, half_dt_curr
        int step, stage

    seed_xoshiro256(&rng_state, seed)
    bm_state.has_spare = 0
    y = x0
    t_particle = 0.0
    stage = 0
    rt_out[0] = -1.0
    choice_out[0] = 0

    for step in range(max_steps):
        dt_curr = T - t_particle
        if dt_curr <= 0.0:
            break
        if dt_curr > dt:
            dt_curr = dt
        sqrt_dt_curr = sqrt(dt_curr)
        half_dt_curr = 0.5 * dt_curr
        z = box_muller_next(&rng_state, &bm_state)
        y = y + mu_array_data[trial_idx, stage] * dt_curr + sigma_array_data[trial_idx, stage] * sqrt_dt_curr * z
        t_particle = t_particle + dt_curr

        # Boundary check with SAME stage used for drift (before advancing)
        upper = ub_array_data[trial_idx, stage] + b1_array_data[trial_idx, stage] * (t_particle - node_array_data[trial_idx, stage])
        lower = lb_array_data[trial_idx, stage] + b2_array_data[trial_idx, stage] * (t_particle - node_array_data[trial_idx, stage])
        if y >= upper:
            rt_out[0] = t_particle - half_dt_curr
            choice_out[0] = 1
            break
        elif y <= lower:
            rt_out[0] = t_particle - half_dt_curr
            choice_out[0] = -1
            break

        # Advance stage for NEXT iteration
        while stage + 1 < d and t_particle >= node_array_data[trial_idx, stage + 1]:
            stage = stage + 1

    x_final_out[0] = y


def simulate_homog_ddm_fpt(
    double[::1] drift_vals,
    double[::1] diffusion_vals,
    double[::1] upper_vals,
    double[::1] lower_vals,
    double[::1] x0_data,
    double dt,
    int max_steps,
    double T,
    uint64_t[::1] trial_seeds,
    int n_threads=1,
):
    """Simulate a batch of DDM trials with precomputed drift/diffusion/boundaries.

    Uses inline C-level RNG (xoshiro256++ / Box-Muller) and OpenMP parallelism.

    Parameters
    ----------
    drift_vals, diffusion_vals, upper_vals, lower_vals : (max_steps,) float64
        Precomputed arrays evaluated on the time grid.
    x0_data : (n_trials,) float64
        Starting point per trial.
    dt : double
        Euler-Maruyama time step.
    max_steps : int
        Number of time steps.
    trial_seeds : (n_trials,) uint64
        Per-trial PRNG seeds (xoshiro256++ seeded via SplitMix64).
    n_threads : int
        Number of OpenMP threads (1 = serial).

    Returns
    -------
    rt_out, choice_out, x_final_out
    """
    cdef:
        int n_trials = trial_seeds.shape[0]
        int trial
    rt_out = np.empty(n_trials, dtype=np.float64)
    choice_out = np.empty(n_trials, dtype=np.int32)
    x_final_out = np.empty(n_trials, dtype=np.float64)

    cdef double[::1] rt_view = rt_out
    cdef int[::1] choice_view = choice_out
    cdef double[::1] x_final_view = x_final_out

    for trial in prange(n_trials, nogil=True, num_threads=n_threads, schedule='dynamic'):
        _run_homog_trial(
            drift_vals, diffusion_vals, upper_vals, lower_vals,
            x0_data[trial], dt, max_steps, T,
            trial_seeds[trial],
            &rt_view[trial], &choice_view[trial], &x_final_view[trial],
        )

    return rt_out, choice_out, x_final_out


def simulate_heterog_multistage_fpt(
    double[:, ::1] mu_array_data,
    double[:, ::1] sigma_array_data,
    double[:, ::1] node_array_data,
    int[::1] d_data,
    double[:, ::1] ub_array_data,
    double[:, ::1] b1_array_data,
    double[:, ::1] lb_array_data,
    double[:, ::1] b2_array_data,
    double[::1] x0_data,
    double dt,
    double T,
    uint64_t[::1] trial_seeds,
    int n_threads=1,
):
    """Simulate a batch of heterogeneous multi-stage DDM trials.

    This is the most general simulator: every trial can have different
    per-stage drift, diffusion, boundary intercepts, slopes, and stage
    onset times.  All arrays are padded to ``max_d`` columns; inactive
    stages should be zero-filled.

    Delegates each trial to ``_run_heterog_trial`` inside an OpenMP
    ``prange`` loop.

    Parameters
    ----------
    mu_array_data : (n_trials, max_d) float64
        Per-stage drift rates.
    sigma_array_data : (n_trials, max_d) float64
        Per-stage diffusion coefficients.
    node_array_data : (n_trials, max_d) float64
        Per-stage onset times (``node_array_data[:, 0]`` should be 0).
    d_data : (n_trials,) int32
        Number of active stages per trial.
    ub_array_data, lb_array_data : (n_trials, max_d) float64
        Upper/lower boundary intercepts at the start of each stage.
    b1_array_data, b2_array_data : (n_trials, max_d) float64
        Upper/lower boundary slopes within each stage.
    x0_data : (n_trials,) float64
        Starting position per trial.
    dt : double
        Euler-Maruyama time step.
    T : double
        Maximum trial duration.
    trial_seeds : (n_trials,) uint64
        Per-trial PRNG seeds.
    n_threads : int
        Number of OpenMP threads (1 = serial).

    Returns
    -------
    rt_out : (n_trials,) float64
        Reaction times (-1.0 if the trial did not terminate by *T*).
    choice_out : (n_trials,) int32
        +1 (upper), -1 (lower), or 0 (no crossing).
    x_final_out : (n_trials,) float64
        Final particle position.
    """
    cdef:
        int n_trials = mu_array_data.shape[0]
        int max_steps
        int trial

    max_steps = int(np.ceil(T / dt)) if T > 0.0 else 0
    rt_out = np.empty(n_trials, dtype=np.float64)
    choice_out = np.empty(n_trials, dtype=np.int32)
    x_final_out = np.empty(n_trials, dtype=np.float64)

    cdef double[::1] rt_view = rt_out
    cdef int[::1] choice_view = choice_out
    cdef double[::1] x_final_view = x_final_out

    if T <= 0.0:
        rt_out.fill(-1.0)
        choice_out.fill(0)
        x_final_out[:] = np.asarray(x0_data)
        return rt_out, choice_out, x_final_out

    for trial in prange(n_trials, nogil=True, num_threads=n_threads, schedule='dynamic'):
        _run_heterog_trial(
            mu_array_data, sigma_array_data, node_array_data,
            d_data[trial],
            ub_array_data, b1_array_data, lb_array_data, b2_array_data,
            trial, x0_data[trial], dt, max_steps, T,
            trial_seeds[trial],
            &rt_view[trial], &choice_view[trial], &x_final_view[trial],
        )

    return rt_out, choice_out, x_final_out


def _simulate_addm_fpt(
    double[:, ::1] mu_array_data,
    double[:, ::1] sacc_array_data,
    int[::1] d_data,
    double sigma,
    double a,
    double b,
    double[::1] x0_data,
    double dt,
    double T,
    uint64_t[::1] trial_seeds,
    int n_threads=1,
):
    """Simulate a batch of aDDM trials via the general multi-stage simulator.
    Internal function that takes a pre-built aDDM mu_array_data from eta, kappa, r1_data, r2_data, flag_data.

    Convenience wrapper that expands the compact aDDM parameterisation
    (scalar ``sigma``, ``a``, ``b``) into full per-stage arrays and
    delegates to ``simulate_heterog_multistage_fpt``.  Boundary intercepts
    are derived from ``a``, ``b``, and the saccade onset times as
    ``ub = a - b * sacc``, ``lb = -(a - b * sacc)``.

    Parameters
    ----------
    mu_array_data : (n_trials, max_d) float64
        Pre-built alternating drift-rate array (from ``_build_addm_mu_array_data``).
    sacc_array_data : (n_trials, max_d) float64
        Saccade onset times (column 0 should be 0).
    d_data : (n_trials,) int32
        Number of fixation stages per trial.
    sigma : double
        Diffusion coefficient (constant across stages).
    a : double
        Boundary intercept at t=0.
    b : double
        Boundary collapse slope (>= 0).
    x0_data : (n_trials,) float64
        Starting position per trial.
    dt : double
        Euler-Maruyama time step.
    T : double
        Maximum trial duration.
    trial_seeds : (n_trials,) uint64
        Per-trial PRNG seeds.
    n_threads : int
        Number of OpenMP threads (1 = serial).

    Returns
    -------
    rt_out, choice_out, x_final_out
        Same as ``simulate_heterog_multistage_fpt``.
    """
    cdef int n_trials = mu_array_data.shape[0]
    cdef int max_d = mu_array_data.shape[1]

    sigma_array_data = np.full((n_trials, max_d), sigma, dtype=np.float64)
    b1_array_data = np.full((n_trials, max_d), -b, dtype=np.float64)
    b2_array_data = np.full((n_trials, max_d), b, dtype=np.float64)

    sacc_np = np.asarray(sacc_array_data)
    ub_array_data = np.ascontiguousarray(a - b * sacc_np, dtype=np.float64)
    lb_array_data = np.ascontiguousarray(-(a - b * sacc_np), dtype=np.float64)

    return simulate_heterog_multistage_fpt(
        mu_array_data, sigma_array_data, sacc_np,
        d_data,
        ub_array_data, b1_array_data,
        lb_array_data, b2_array_data,
        x0_data,
        dt, T,
        trial_seeds,
        n_threads,
    )
