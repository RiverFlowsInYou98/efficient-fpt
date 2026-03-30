# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython-accelerated batch likelihood computation for ADDM and TADA models.

Provides ``compute_addm_likelihoods``, ``compute_addm_mean_nll``,
``compute_addm_sum_nll``,
``compute_tada_likelihoods``, and ``compute_tada_mean_nll``.

All functions accept an ``n_threads`` parameter: 1 for serial execution,
>1 or -1 (all available) for OpenMP parallel execution.
"""

import warnings
import numpy as np
cimport numpy as np
from libc.math cimport log, isnan, isinf
from cython cimport boundscheck, wraparound, cdivision
from cython.parallel import prange

from .single_stage cimport fptd_single
from .multi_stage cimport _compute_addm_fptd_core
from .multi_stage import _get_quad_data, compute_addm_fptd
from ..addm_helpers import _build_addm_mu_array_data

include "_defaults.pxi"

cdef extern from "omp.h":
    int omp_get_max_threads()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef inline np.uint8_t _classify_likelihood(double likelihood) noexcept nogil:
    if likelihood == 0.0:
        return 1
    if likelihood > 0.0 and not isnan(likelihood) and not isinf(likelihood):
        return 0
    return 2


def _warn_skipped_trial(Py_ssize_t idx, np.uint8_t reason):
    """Emit a deterministic skipped-trial warning for one trial."""
    if reason == 1:
        warnings.warn(
            f"trial {idx} outputs 0 likelihood, skipped",
            RuntimeWarning,
            stacklevel=2,
        )
    elif reason == 2:
        warnings.warn(
            f"trial {idx} outputs invalid likelihood, skipped",
            RuntimeWarning,
            stacklevel=2,
        )
    elif reason == 3:
        warnings.warn(
            f"trial {idx} has nonpositive rt, skipped",
            RuntimeWarning,
            stacklevel=2,
        )


def _warn_skipped_trials(np.ndarray[np.uint8_t, ndim=1] reason_data):
    """Emit deterministic skipped-trial warnings in ascending trial order."""
    cdef Py_ssize_t idx
    for idx in range(reason_data.shape[0]):
        _warn_skipped_trial(idx, reason_data[idx])


# ---------------------------------------------------------------------------
# ADDM likelihoods
# ---------------------------------------------------------------------------

cpdef np.ndarray[double, ndim=1] compute_addm_likelihoods(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order=DEFAULT_QUADRATURE_ORDER, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False,
):
    """Per-trial ADDM likelihoods. Returns 1-D array; 0.0 for invalid trials.

    Parameters
    ----------
    n_threads : int
        1 for serial, >1 for that many threads, -1 for all available.
    """
    cdef:
        int n
        int num_data = len(rt_data)
        int max_d = sacc_array_data.shape[1]
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = _build_addm_mu_array_data(
            eta, kappa, r1_data, r2_data, flag_data, d_data, max_d,
        )
        double[:, :] mu_view = mu_array_data
        double[:, :] sacc_view = sacc_array_data
    if n_threads == 1:
        # Serial path
        for n in range(num_data):
            likelihood = compute_addm_fptd(
                rt_data[n],
                choice_data[n],
                eta,
                kappa,
                sigma,
                a,
                b,
                x0,
                r1_data[n],
                r2_data[n],
                flag_data[n],
                sacc_view[n],
                d_data[n],
                order=order,
                trunc_num=trunc_num,
                threshold=threshold,
                log_space=log_space,
            )
            if _classify_likelihood(likelihood) == 0:
                likelihoods[n] = likelihood
            else:
                likelihoods[n] = 0.0
    else:
        # Parallel path (nogil)
        _compute_addm_likelihoods_parallel(
            rt_data, choice_data, mu_array_data, sacc_array_data, d_data,
            sigma, a, b, x0,
            order, trunc_num, threshold, n_threads, log_space,
            likelihoods,
        )

    return likelihoods


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void _compute_addm_likelihoods_parallel(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    np.ndarray[double, ndim=2] mu_array_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    double sigma, double a, double b, double x0,
    int order, int trunc_num, double threshold, int n_threads, bint log_space,
    np.ndarray[double, ndim=1] likelihoods,
):
    """Parallel ADDM likelihoods using prange."""
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        double[:, :] mu_view = mu_array_data
        double[:, :] sacc_view = sacc_array_data
        double[:] lik_view = likelihoods

    x_np, w_np = _get_quad_data(order)
    cdef double[:] x_ref = x_np
    cdef double[:] w_ref = w_np

    if n_threads <= 0:
        n_threads = omp_get_max_threads()

    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=n_threads):
        likelihood = _compute_addm_fptd_core(
            rt_data[n], d_data[n], mu_view[n], sacc_view[n],
            sigma, a, b, x0, choice_data[n],
            order, x_ref, w_ref, trunc_num, threshold,
            log_space,
        )
        if _classify_likelihood(likelihood) == 0:
            lik_view[n] = likelihood
        else:
            lik_view[n] = 0.0


# ---------------------------------------------------------------------------
# ADDM mean NLL
# ---------------------------------------------------------------------------

cpdef double compute_addm_mean_nll(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order=DEFAULT_QUADRATURE_ORDER, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False, bint warn=True,
):
    """Mean negative log-likelihood for ADDM trials.

    Dispatches to serial or parallel based on *n_threads*.
    """
    cdef:
        int n
        int num_data = len(rt_data)
        int num_data_effective = 0
        double total_loss = 0.0
        np.uint8_t reason
        np.ndarray[np.uint8_t, ndim=1] reason_data = np.zeros(num_data, dtype=np.uint8)

    cdef np.ndarray[double, ndim=1] likelihoods = compute_addm_likelihoods(
        rt_data, choice_data,
        eta, kappa, sigma, a, b, x0,
        r1_data, r2_data, flag_data,
        sacc_array_data, d_data,
        order=order, trunc_num=trunc_num, threshold=threshold,
        n_threads=n_threads, log_space=log_space,
    )

    for n in range(num_data):
        if likelihoods[n] > 0.0:
            total_loss += -log(likelihoods[n])
            num_data_effective += 1
        elif likelihoods[n] == 0.0:
            reason_data[n] = 1
        else:
            reason_data[n] = 2

    if warn:
        _warn_skipped_trials(reason_data)
    if num_data_effective == 0:
        return float("nan")
    return total_loss / num_data_effective


cpdef double compute_addm_sum_nll(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order=DEFAULT_QUADRATURE_ORDER, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False, bint warn=True,
):
    """Summed negative log-likelihood for ADDM trials."""
    cdef:
        int n
        int num_data = len(rt_data)
        double total_loss = 0.0
        np.ndarray[np.uint8_t, ndim=1] reason_data = np.zeros(num_data, dtype=np.uint8)

    cdef np.ndarray[double, ndim=1] likelihoods = compute_addm_likelihoods(
        rt_data, choice_data,
        eta, kappa, sigma, a, b, x0,
        r1_data, r2_data, flag_data,
        sacc_array_data, d_data,
        order=order, trunc_num=trunc_num, threshold=threshold,
        n_threads=n_threads, log_space=log_space,
    )

    for n in range(num_data):
        if likelihoods[n] > 0.0:
            total_loss += -log(likelihoods[n])
        elif likelihoods[n] == 0.0:
            reason_data[n] = 1
        else:
            reason_data[n] = 2

    if warn:
        _warn_skipped_trials(reason_data)
    if np.all(reason_data != 0):
        return float("nan")
    return total_loss


cpdef double compute_addm_nll(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order=DEFAULT_QUADRATURE_ORDER, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False, str reduce="mean", bint warn=True,
):
    """Unified negative log-likelihood for ADDM trials.

    Parameters
    ----------
    reduce : str
        ``"mean"`` (default) or ``"sum"``.
    warn : bool
        If True, emit warnings for skipped trials.

    Signature matches the JAX ``compute_addm_nll`` (except for ``n_threads``).
    """
    if reduce == "mean":
        return compute_addm_mean_nll(
            rt_data, choice_data,
            eta, kappa, sigma, a, b, x0,
            r1_data, r2_data, flag_data,
            sacc_array_data, d_data,
            order=order, trunc_num=trunc_num, threshold=threshold,
            n_threads=n_threads, log_space=log_space, warn=warn,
        )
    elif reduce == "sum":
        return compute_addm_sum_nll(
            rt_data, choice_data,
            eta, kappa, sigma, a, b, x0,
            r1_data, r2_data, flag_data,
            sacc_array_data, d_data,
            order=order, trunc_num=trunc_num, threshold=threshold,
            n_threads=n_threads, log_space=log_space, warn=warn,
        )
    else:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")


# ---------------------------------------------------------------------------
# TADA likelihoods
# ---------------------------------------------------------------------------

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_tada_likelihoods(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1,
):
    """Per-trial TADA likelihoods. Returns 1-D array; 0.0 for invalid trials.

    Computes a time-averaged drift per trial, then evaluates single-stage FPTD.
    """
    cdef:
        int n, i, L
        double likelihood, mu_sum
        int num_data = len(rt_data)
        int max_d = sacc_array_data.shape[1]
        np.ndarray[double, ndim=2] mu_array_data = _build_addm_mu_array_data(
            eta, kappa, r1_data, r2_data, flag_data, d_data, max_d,
        )
        double[:, :] mu_view = mu_array_data
        double[:, :] sacc_view = sacc_array_data
        np.ndarray[double, ndim=1] mu_tada_data = np.zeros(num_data, dtype=np.float64)
        np.ndarray[double, ndim=1] likelihoods = np.zeros(num_data, dtype=np.float64)
        np.ndarray[np.uint8_t, ndim=1] reason_data = np.zeros(num_data, dtype=np.uint8)
        double[:] mu_tada_view = mu_tada_data
        double[:] lik_view = likelihoods
        np.uint8_t[:] reason_view = reason_data

    # Compute time-averaged drift (requires GIL for array access safety)
    for n in range(num_data):
        L = d_data[n]
        if rt_data[n] > 0.0:
            mu_sum = 0.0
            for i in range(L - 1):
                mu_sum += mu_view[n, i] * (sacc_view[n, i + 1] - sacc_view[n, i])
            mu_sum += mu_view[n, L - 1] * (rt_data[n] - sacc_view[n, L - 1])
            mu_tada_view[n] = mu_sum / rt_data[n]
        else:
            reason_view[n] = 3

    if n_threads <= 0 or n_threads > 1:
        if n_threads <= 0:
            n_threads = omp_get_max_threads()
        for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=n_threads):
            if reason_view[n] != 0:
                lik_view[n] = 0.0
                continue
            likelihood = fptd_single(
                rt_data[n], mu_tada_view[n], sigma, a, -b, -a, b,
                x0, choice_data[n], trunc_num, threshold, True,
            )
            if _classify_likelihood(likelihood) == 0:
                lik_view[n] = likelihood
            else:
                lik_view[n] = 0.0
                reason_view[n] = _classify_likelihood(likelihood)
    else:
        # Serial path
        for n in range(num_data):
            if reason_view[n] != 0:
                lik_view[n] = 0.0
                continue
            likelihood = fptd_single(
                rt_data[n], mu_tada_view[n], sigma, a, -b, -a, b,
                x0, choice_data[n], trunc_num, threshold, True,
            )
            if _classify_likelihood(likelihood) == 0:
                lik_view[n] = likelihood
            else:
                lik_view[n] = 0.0
                reason_view[n] = _classify_likelihood(likelihood)

    return likelihoods


# ---------------------------------------------------------------------------
# TADA mean NLL
# ---------------------------------------------------------------------------

cpdef double compute_tada_mean_nll(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint warn=True,
):
    """Mean negative log-likelihood using time-averaged drift approximation.

    Dispatches to serial or parallel based on *n_threads*.
    """
    cdef:
        int n
        int num_data = len(rt_data)
        int num_data_effective = 0
        double total_loss = 0.0
        np.ndarray[np.uint8_t, ndim=1] reason_data = np.zeros(num_data, dtype=np.uint8)

    # Pre-flag nonpositive-rt trials before computing likelihoods
    for n in range(num_data):
        if rt_data[n] <= 0.0:
            reason_data[n] = 3

    cdef np.ndarray[double, ndim=1] likelihoods = compute_tada_likelihoods(
        rt_data, choice_data,
        eta, kappa, sigma, a, b, x0,
        r1_data, r2_data, flag_data,
        sacc_array_data, d_data,
        trunc_num=trunc_num, threshold=threshold,
        n_threads=n_threads,
    )

    for n in range(num_data):
        if reason_data[n] != 0:
            # Already flagged (e.g. nonpositive rt)
            continue
        if likelihoods[n] > 0.0:
            total_loss += -log(likelihoods[n])
            num_data_effective += 1
        elif likelihoods[n] == 0.0:
            reason_data[n] = 1
        else:
            reason_data[n] = 2

    if warn:
        _warn_skipped_trials(reason_data)
    if num_data_effective == 0:
        return float("nan")
    return total_loss / num_data_effective
