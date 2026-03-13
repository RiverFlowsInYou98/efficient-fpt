# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt


def simulate_addm_batch_cy(
    double[:, ::1] mu_data,
    double[:, ::1] sacc_data,
    int[::1] d_data,
    double sigma,
    double a,
    double b,
    double x0,
    double dt,
    double max_t,
    double[:, ::1] gaussian_data,
):
    """Simulate a batch of aDDM trials using Euler-Maruyama.

    All random draws are pre-generated in NumPy (Ziggurat) and passed in as
    ``gaussian_data``.  The inner loop is pure C -- no Python calls, no GIL.

    Parameters
    ----------
    mu_data : (n_trials, max_d) C-contiguous float64
        Drift rate per fixation stage, zero-padded.
    sacc_data : (n_trials, max_d) C-contiguous float64
        Saccade onset times per stage, zero-padded.
    d_data : (n_trials,) C-contiguous int32
        Number of fixation stages per trial.
    sigma, a, b, x0, dt, max_t : double
        Diffusion coeff, boundary intercept/slope, start point, timestep, deadline.
    gaussian_data : (n_trials, max_steps) C-contiguous float64
        Pre-generated standard normal draws.

    Returns
    -------
    rt_out : ndarray (n_trials,) float64
        Reaction times.  -1.0 for trials that did not terminate.
    choice_out : ndarray (n_trials,) int32
        +1 (upper boundary) or -1 (lower boundary).  0 if no crossing.
    """
    cdef:
        int n_trials = mu_data.shape[0]
        int max_steps = gaussian_data.shape[1]
        double sigma_sqrt_dt = sigma * sqrt(dt)
        double half_dt = 0.5 * dt
        int trial, stage, step, d
        double y, t_particle, upper

    rt_out = np.empty(n_trials, dtype=np.float64)
    choice_out = np.empty(n_trials, dtype=np.int32)

    cdef double[::1] rt_view = rt_out
    cdef int[::1] choice_view = choice_out

    with nogil:
        for trial in range(n_trials):
            d = d_data[trial]
            y = x0
            t_particle = 0.0
            stage = 0

            rt_view[trial] = -1.0
            choice_view[trial] = 0

            for step in range(max_steps):
                y += mu_data[trial, stage] * dt + sigma_sqrt_dt * gaussian_data[trial, step]
                t_particle += dt

                while stage + 1 < d and t_particle >= sacc_data[trial, stage + 1]:
                    stage += 1

                upper = a - b * t_particle
                if y >= upper:
                    rt_view[trial] = t_particle - half_dt
                    choice_view[trial] = 1
                    break
                elif y <= -upper:
                    rt_view[trial] = t_particle - half_dt
                    choice_view[trial] = -1
                    break

    return rt_out, choice_out
