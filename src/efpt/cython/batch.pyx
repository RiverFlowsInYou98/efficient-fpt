# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython-accelerated batch log-likelihood computation for ADDM and TADA models.

Provides ``compute_addm_loglikelihoods``, ``compute_addm_mean_nll``,
``compute_addm_sum_nll``,
``compute_tada_loglikelihoods``, and ``compute_tada_mean_nll``.

All functions accept an ``n_threads`` parameter: 1 for serial execution,
>1 or -1 (all available) for OpenMP parallel execution.
"""

import warnings
import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf, INFINITY
from cython cimport boundscheck, wraparound, cdivision
from cython.parallel import prange

from .single_stage cimport fptd_single
from .multi_stage cimport _compute_addm_logfptd_core
from .multi_stage import _get_quad_data, compute_addm_logfptd
from .utils cimport positive_log
from ..addm_helpers import _build_addm_mu_array_data

include "_defaults.pxi"

cdef extern from "omp.h":
    int omp_get_max_threads()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef inline np.uint8_t _classify_loglikelihood(double loglikelihood) noexcept nogil:
    if loglikelihood == -INFINITY:
        return 1
    if not isnan(loglikelihood) and not isinf(loglikelihood):
        return 0
    return 2


def _warn_bad_trial(Py_ssize_t idx, np.uint8_t reason):
    """Emit a deterministic bad-trial warning for one trial."""
    if reason == 1:
        warnings.warn(
            f"trial {idx} outputs -inf log-likelihood",
            RuntimeWarning,
            stacklevel=2,
        )
    elif reason == 2:
        warnings.warn(
            f"trial {idx} outputs invalid log-likelihood",
            RuntimeWarning,
            stacklevel=2,
        )
    elif reason == 3:
        warnings.warn(
            f"trial {idx} has nonpositive rt, skipped",
            RuntimeWarning,
            stacklevel=2,
        )


def _warn_bad_trials(np.ndarray[np.uint8_t, ndim=1] reason_data):
    """Emit deterministic bad-trial warnings in ascending trial order."""
    cdef Py_ssize_t idx
    for idx in range(reason_data.shape[0]):
        _warn_bad_trial(idx, reason_data[idx])


def _reduce_loglikelihoods_to_nll(
    np.ndarray[double, ndim=1] loglikelihoods,
    str reduce,
    str invalid_policy,
    bint warn,
    np.ndarray[np.uint8_t, ndim=1] reason_data=None,
):
    """Shared reducer for log-likelihood vectors with deterministic warnings."""
    cdef:
        Py_ssize_t n
        Py_ssize_t num_data = loglikelihoods.shape[0]
        Py_ssize_t num_data_effective = 0
        double total_loss = 0.0
        bint saw_neginf = False
        bint saw_invalid = False

    if reason_data is None:
        reason_data = np.zeros(num_data, dtype=np.uint8)

    for n in range(num_data):
        if reason_data[n] != 0:
            continue
        if _classify_loglikelihood(loglikelihoods[n]) == 0:
            total_loss += -loglikelihoods[n]
            num_data_effective += 1
        elif loglikelihoods[n] == -INFINITY:
            reason_data[n] = 1
            saw_neginf = True
        else:
            reason_data[n] = 2
            saw_invalid = True

    if warn:
        _warn_bad_trials(reason_data)
    if invalid_policy == "inf":
        if saw_invalid:
            return float("nan")
        if saw_neginf:
            return float("inf")
    elif invalid_policy != "warn":
        raise ValueError(
            f"invalid_policy must be 'inf' or 'warn', got {invalid_policy!r}"
        )

    if num_data_effective == 0:
        return float("nan")
    if reduce == "sum":
        return total_loss
    if reduce == "mean":
        return total_loss / num_data_effective
    raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")


# ---------------------------------------------------------------------------
# ADDM log-likelihoods
# ---------------------------------------------------------------------------

cpdef np.ndarray[double, ndim=1] compute_addm_loglikelihoods(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order_mid=DEFAULT_MID_QUAD_ORDER,
    int order_last=DEFAULT_LAST_QUAD_ORDER,
    object order=None,
    int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False,
):
    """Per-trial ADDM log-likelihoods.

    Parameters
    ----------
    n_threads : int
        1 for serial, >1 for that many threads, -1 for all available.
    """
    cdef:
        int n
        int num_data = len(rt_data)
        int max_d = sacc_array_data.shape[1]
        double loglikelihood
        np.ndarray[double, ndim=1] loglikelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = _build_addm_mu_array_data(
            eta, kappa, r1_data, r2_data, flag_data, d_data, max_d,
        )
        double[:, :] mu_view = mu_array_data
        double[:, :] sacc_view = sacc_array_data
    if order is not None:
        if order_mid != DEFAULT_MID_QUAD_ORDER or order_last != DEFAULT_LAST_QUAD_ORDER:
            raise ValueError(
                "pass either legacy order or split order_mid/order_last, not both"
            )
        order_mid = int(order)
        order_last = int(order)
    if n_threads == 1:
        # Serial path
        for n in range(num_data):
            loglikelihood = compute_addm_logfptd(
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
                order_mid=order_mid,
                order_last=order_last,
                trunc_num=trunc_num,
                threshold=threshold,
                log_space=log_space,
            )
            loglikelihoods[n] = loglikelihood
    else:
        # Parallel path (nogil)
        _compute_addm_loglikelihoods_parallel(
            rt_data, choice_data, mu_array_data, sacc_array_data, d_data,
            sigma, a, b, x0,
            order_mid, order_last, trunc_num, threshold, n_threads, log_space,
            loglikelihoods,
        )

    return loglikelihoods


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void _compute_addm_loglikelihoods_parallel(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    np.ndarray[double, ndim=2] mu_array_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    double sigma, double a, double b, double x0,
    int order_mid, int order_last, int trunc_num, double threshold, int n_threads, bint log_space,
    np.ndarray[double, ndim=1] loglikelihoods,
):
    """Parallel ADDM log-likelihoods using prange."""
    cdef:
        int n
        int num_data = len(rt_data)
        double loglikelihood
        double[:, :] mu_view = mu_array_data
        double[:, :] sacc_view = sacc_array_data
        double[:] loglik_view = loglikelihoods

    x_mid_np, w_mid_np = _get_quad_data(order_mid)
    x_last_np, w_last_np = _get_quad_data(order_last)
    cdef double[:] x_ref_mid = x_mid_np
    cdef double[:] w_ref_mid = w_mid_np
    cdef double[:] x_ref_last = x_last_np
    cdef double[:] w_ref_last = w_last_np

    if n_threads <= 0:
        n_threads = omp_get_max_threads()

    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=n_threads):
        loglikelihood = _compute_addm_logfptd_core(
            rt_data[n], d_data[n], mu_view[n], sacc_view[n],
            sigma, a, b, x0, choice_data[n],
            order_mid, x_ref_mid, w_ref_mid, order_last, x_ref_last, w_ref_last,
            trunc_num, threshold,
            log_space,
        )
        loglik_view[n] = loglikelihood


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
    int order_mid=DEFAULT_MID_QUAD_ORDER,
    int order_last=DEFAULT_LAST_QUAD_ORDER,
    object order=None,
    int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False, str invalid_policy="inf", bint warn=True,
):
    """Mean negative log-likelihood for ADDM trials.

    Dispatches to serial or parallel based on *n_threads*.
    """
    cdef np.ndarray[double, ndim=1] loglikelihoods = compute_addm_loglikelihoods(
        rt_data, choice_data,
        eta, kappa, sigma, a, b, x0,
        r1_data, r2_data, flag_data,
        sacc_array_data, d_data,
        order_mid=order_mid, order_last=order_last, order=order,
        trunc_num=trunc_num, threshold=threshold,
        n_threads=n_threads, log_space=log_space,
    )
    return _reduce_loglikelihoods_to_nll(
        loglikelihoods,
        reduce="mean",
        invalid_policy=invalid_policy,
        warn=warn,
    )


cpdef double compute_addm_sum_nll(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order_mid=DEFAULT_MID_QUAD_ORDER,
    int order_last=DEFAULT_LAST_QUAD_ORDER,
    object order=None,
    int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False, str invalid_policy="inf", bint warn=True,
):
    """Summed negative log-likelihood for ADDM trials."""
    cdef np.ndarray[double, ndim=1] loglikelihoods = compute_addm_loglikelihoods(
        rt_data, choice_data,
        eta, kappa, sigma, a, b, x0,
        r1_data, r2_data, flag_data,
        sacc_array_data, d_data,
        order_mid=order_mid, order_last=order_last, order=order,
        trunc_num=trunc_num, threshold=threshold,
        n_threads=n_threads, log_space=log_space,
    )
    return _reduce_loglikelihoods_to_nll(
        loglikelihoods,
        reduce="sum",
        invalid_policy=invalid_policy,
        warn=warn,
    )


cpdef double compute_addm_nll(
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    double eta, double kappa, double sigma, double a, double b, double x0,
    np.ndarray[double, ndim=1] r1_data,
    np.ndarray[double, ndim=1] r2_data,
    np.ndarray[int, ndim=1] flag_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int order_mid=DEFAULT_MID_QUAD_ORDER,
    int order_last=DEFAULT_LAST_QUAD_ORDER,
    object order=None,
    int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD,
    int n_threads=1, bint log_space=False, str reduce="mean", str invalid_policy="inf", bint warn=True,
):
    """Unified negative log-likelihood for ADDM trials.

    Parameters
    ----------
    reduce : str
        ``"mean"`` (default) or ``"sum"``.
    warn : bool
        If True, emit warnings for bad trials.

    Signature matches the JAX ``compute_addm_nll`` (except for ``n_threads``).
    """
    if reduce == "mean":
        return compute_addm_mean_nll(
            rt_data, choice_data,
            eta, kappa, sigma, a, b, x0,
            r1_data, r2_data, flag_data,
            sacc_array_data, d_data,
            order_mid=order_mid, order_last=order_last, order=order,
            trunc_num=trunc_num, threshold=threshold,
            n_threads=n_threads, log_space=log_space, invalid_policy=invalid_policy, warn=warn,
        )
    elif reduce == "sum":
        return compute_addm_sum_nll(
            rt_data, choice_data,
            eta, kappa, sigma, a, b, x0,
            r1_data, r2_data, flag_data,
            sacc_array_data, d_data,
            order_mid=order_mid, order_last=order_last, order=order,
            trunc_num=trunc_num, threshold=threshold,
            n_threads=n_threads, log_space=log_space, invalid_policy=invalid_policy, warn=warn,
        )
    else:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")


# ---------------------------------------------------------------------------
# TADA log-likelihoods
# ---------------------------------------------------------------------------

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_tada_loglikelihoods(
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
    """Per-trial TADA log-likelihoods.

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
        np.ndarray[double, ndim=1] loglikelihoods = np.zeros(num_data, dtype=np.float64)
        np.ndarray[np.uint8_t, ndim=1] reason_data = np.zeros(num_data, dtype=np.uint8)
        double[:] mu_tada_view = mu_tada_data
        double[:] loglik_view = loglikelihoods
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
                loglik_view[n] = -INFINITY
                continue
            likelihood = fptd_single(
                rt_data[n], mu_tada_view[n], sigma, a, -b, -a, b,
                x0, choice_data[n], trunc_num, threshold, True,
            )
            loglik_view[n] = positive_log(likelihood)
            if _classify_loglikelihood(loglik_view[n]) != 0:
                reason_view[n] = _classify_loglikelihood(loglik_view[n])
    else:
        # Serial path
        for n in range(num_data):
            if reason_view[n] != 0:
                loglik_view[n] = -INFINITY
                continue
            likelihood = fptd_single(
                rt_data[n], mu_tada_view[n], sigma, a, -b, -a, b,
                x0, choice_data[n], trunc_num, threshold, True,
            )
            loglik_view[n] = positive_log(likelihood)
            if _classify_loglikelihood(loglik_view[n]) != 0:
                reason_view[n] = _classify_loglikelihood(loglik_view[n])

    return loglikelihoods


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
    int n_threads=1, str invalid_policy="inf", bint warn=True,
):
    """Mean negative log-likelihood using time-averaged drift approximation.

    Dispatches to serial or parallel based on *n_threads*.
    """
    cdef:
        int n
        int num_data = len(rt_data)
        np.ndarray[np.uint8_t, ndim=1] reason_data = np.zeros(num_data, dtype=np.uint8)

    # Pre-flag nonpositive-rt trials before computing likelihoods
    for n in range(num_data):
        if rt_data[n] <= 0.0:
            reason_data[n] = 3

    cdef np.ndarray[double, ndim=1] loglikelihoods = compute_tada_loglikelihoods(
        rt_data, choice_data,
        eta, kappa, sigma, a, b, x0,
        r1_data, r2_data, flag_data,
        sacc_array_data, d_data,
        trunc_num=trunc_num, threshold=threshold,
        n_threads=n_threads,
    )
    return _reduce_loglikelihoods_to_nll(
        loglikelihoods,
        reduce="mean",
        invalid_policy=invalid_policy,
        warn=warn,
        reason_data=reason_data,
    )
