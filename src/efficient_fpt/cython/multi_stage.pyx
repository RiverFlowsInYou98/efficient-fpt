# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython-accelerated multi-stage FPT density computation.

Provides ``compute_addm_fptd`` and ``compute_heterog_multistage_fptd`` (parameterized by
quadrature *order*).  Batch likelihood helpers live in ``batch.pyx``.
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, M_PI, fabs, log, INFINITY, isnan, isinf
from libc.stdio cimport fprintf, stderr
from cython cimport boundscheck, wraparound, cdivision

from .single_stage cimport fptd_single, q_single
from ..quadrature import lgwt_lookup_table

include "_defaults.pxi"

# ---------------------------------------------------------------------------
# Quadrature data cache
# ---------------------------------------------------------------------------

def _get_quad_data(int order):
    """Return (x_ref, w_ref) arrays on [-1, 1] for the given order."""
    x, w = lgwt_lookup_table(order, -1.0, 1.0)
    return (
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w, dtype=np.float64),
    )

# Maximum quadrature order supported by the stack buffers.
DEF MAX_ORDER = 200


# ---------------------------------------------------------------------------
# Nogil helpers
# ---------------------------------------------------------------------------

cdef double _logsumexp(double* arr, int n) noexcept nogil:
    """Numerically stable log(sum(exp(arr))) for nogil context."""
    cdef double max_val = arr[0]
    cdef double total = 0.0
    cdef int i
    for i in range(1, n):
        if arr[i] > max_val:
            max_val = arr[i]
    if max_val == -INFINITY:
        return max_val
    for i in range(n):
        total += exp(arr[i] - max_val)
    return max_val + log(total)


# ---------------------------------------------------------------------------
# Core ADDM FPTD (unified normal / log-space)
# ---------------------------------------------------------------------------

cdef double _compute_addm_fptd_core(
    double t, int d,
    double[:] mu_array, double[:] sacc_array,
    double sigma, double a, double b, double x0, int bdy,
    int order, double[:] x_ref_in, double[:] w_ref_in,
    int trunc_num, double threshold,
    bint log_space,
) noexcept nogil:
    """GIL-free core: compute ADDM FPTD in normal-space or log-space.

    When *log_space* is True, tracks log(ws*pv) instead of ws*pv to
    prevent underflow in deep (many-stage) models.
    """
    cdef:
        double result = 0.0
        int i, j, n
        double temp, fptd_val, a_curr, T_curr
        double NEG_INF = -INFINITY

        # Stack buffers
        double x_ref[MAX_ORDER]
        double w_ref[MAX_ORDER]
        double xs[MAX_ORDER]
        double ws[MAX_ORDER]
        double xs_prev[MAX_ORDER]
        # Normal-space buffers
        double pv[MAX_ORDER]
        double ws_pv_prev[MAX_ORDER]
        # Log-space buffers
        double log_ws_pv_prev[MAX_ORDER]
        double log_ws_pv_new[MAX_ORDER]
        double log_terms[MAX_ORDER]

    if order > MAX_ORDER:
        fprintf(stderr, "_compute_addm_fptd_core: order=%d exceeds MAX_ORDER=%d, clamping\n", order, MAX_ORDER)
        order = MAX_ORDER

    if d == 1:
        return fptd_single(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold, True)

    for i in range(order):
        x_ref[i] = x_ref_in[i]
        w_ref[i] = w_ref_in[i]

    # --- First stage ---
    for i in range(order):
        xs[i] = x_ref[i] * (a - b * sacc_array[1])
        ws[i] = w_ref[i] * (a - b * sacc_array[1])
        temp = q_single(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold, True)
        xs_prev[i] = xs[i]
        if log_space:
            if ws[i] * temp > 0:
                log_ws_pv_prev[i] = log(ws[i] * temp)
            else:
                log_ws_pv_prev[i] = NEG_INF
        else:
            ws_pv_prev[i] = ws[i] * temp

    # --- Intermediate stages ---
    for n in range(2, d):
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[n])
            ws[i] = w_ref[i] * (a - b * sacc_array[n])
            a_curr = a - b * sacc_array[n - 1]
            T_curr = sacc_array[n] - sacc_array[n - 1]

            if log_space:
                for j in range(order):
                    temp = q_single(xs[i], mu_array[n - 1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold, True)
                    if temp > 0:
                        log_terms[j] = log(temp) + log_ws_pv_prev[j]
                    else:
                        log_terms[j] = NEG_INF
                temp = _logsumexp(log_terms, order)
                if ws[i] > 0:
                    log_ws_pv_new[i] = log(ws[i]) + temp
                else:
                    log_ws_pv_new[i] = NEG_INF
            else:
                pv[i] = 0
                for j in range(order):
                    temp = q_single(xs[i], mu_array[n - 1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold, True)
                    pv[i] += temp * ws_pv_prev[j]

        for i in range(order):
            xs_prev[i] = xs[i]
            if log_space:
                log_ws_pv_prev[i] = log_ws_pv_new[i]
            else:
                ws_pv_prev[i] = ws[i] * pv[i]

    # --- Final stage ---
    a_curr = a - b * sacc_array[d - 1]
    if log_space:
        for i in range(order):
            fptd_val = fptd_single(t - sacc_array[d - 1], mu_array[d - 1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold, True)
            if fptd_val > 0:
                log_terms[i] = log(fptd_val) + log_ws_pv_prev[i]
            else:
                log_terms[i] = NEG_INF
        return exp(_logsumexp(log_terms, order))
    else:
        for i in range(order):
            fptd_val = fptd_single(t - sacc_array[d - 1], mu_array[d - 1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold, True)
            result += fptd_val * ws_pv_prev[i]
        return result


# ---------------------------------------------------------------------------
# Public ADDM FPTD
# ---------------------------------------------------------------------------

cpdef double compute_addm_fptd(
    double rt, int choice,
    double eta, double kappa, double sigma, double a, double b, double x0,
    double r1, double r2, int flag,
    double[:] sacc_array, int d,
    int order=DEFAULT_QUADRATURE_ORDER, int trunc_num=DEFAULT_TRUNC_NUM,
    double threshold=DEFAULT_THRESHOLD,
    bint log_space=False,
):
    """Compute the ADDM likelihood of hitting *choice* at time *rt*.

    Parameters
    ----------
    rt : double
        Time at which to evaluate the FPTD.
    choice : int
        +1 for the upper boundary, -1 for the lower boundary.
    eta, kappa, sigma, a, b, x0 : double
        ADDM parameters.
    r1, r2 : double
        Stimulus ratings.
    flag : int
        0 = fixate item 1 first, 1 = fixate item 2 first.
    sacc_array : memoryview (d,)
        Stage onset times.
    d : int
        Number of valid stages.
    order : int
        Gauss-Legendre quadrature order (default 30).
    trunc_num : int
        Series truncation limit.
    log_space : bool
        If True, use log-space computation to prevent underflow in deep models.
    """
    cdef:
        double[:] x_ref
        double[:] w_ref
        np.ndarray[double, ndim=1] mu_array_np = np.empty(d, dtype=np.float64)
        double[:] mu_array = mu_array_np
        double mu1 = kappa * (r1 - eta * r2)
        double mu2 = kappa * (eta * r1 - r2)
        int idx

    for idx in range(d):
        if flag == 0:
            mu_array[idx] = mu1 if idx % 2 == 0 else mu2
        else:
            mu_array[idx] = mu2 if idx % 2 == 0 else mu1

    x_np, w_np = _get_quad_data(order)
    x_ref = x_np
    w_ref = w_np

    return _compute_addm_fptd_core(
        rt, d, mu_array, sacc_array, sigma, a, b, x0, choice,
        order, x_ref, w_ref, trunc_num, threshold, log_space,
    )


# ---------------------------------------------------------------------------
# Core generalized multi-stage FPTD (unified normal / log-space)
# ---------------------------------------------------------------------------

cdef double _compute_multistage_fptd_core(
    double t, int d,
    double[:] mu_array, double[:] node_array,
    double[:] sigma_array,
    double a1, double[:] b1_array,
    double a2, double[:] b2_array,
    double x0, int bdy,
    int order, double[:] x_ref_in, double[:] w_ref_in,
    int trunc_num, double threshold,
    bint log_space,
) noexcept nogil:
    """GIL-free core: generalized multi-stage FPTD with per-stage sigma and slopes."""
    cdef:
        double result = 0.0
        int i, j, n
        double temp, fptd_val, T_curr
        double ub, lb, ub_prev, lb_prev
        double half_width, center
        double NEG_INF = -INFINITY

        # Stack buffers
        double x_ref[MAX_ORDER]
        double w_ref[MAX_ORDER]
        double xs[MAX_ORDER]
        double ws[MAX_ORDER]
        double xs_prev[MAX_ORDER]
        # Normal-space buffers
        double pv[MAX_ORDER]
        double ws_pv_prev[MAX_ORDER]
        # Log-space buffers
        double log_ws_pv_prev[MAX_ORDER]
        double log_ws_pv_new[MAX_ORDER]
        double log_terms[MAX_ORDER]

    if order > MAX_ORDER:
        fprintf(stderr, "_compute_multistage_fptd_core: order=%d exceeds MAX_ORDER=%d, clamping\n", order, MAX_ORDER)
        order = MAX_ORDER

    if d == 1:
        return fptd_single(t, mu_array[0], sigma_array[0], a1, b1_array[0], a2, b2_array[0], x0, bdy, trunc_num, threshold, True)

    for i in range(order):
        x_ref[i] = x_ref_in[i]
        w_ref[i] = w_ref_in[i]

    ub_prev = a1
    lb_prev = a2

    # --- First stage ---
    T_curr = node_array[1] - node_array[0]
    ub = ub_prev + b1_array[0] * T_curr
    lb = lb_prev + b2_array[0] * T_curr
    half_width = (ub - lb) / 2.0
    center = (ub + lb) / 2.0

    for i in range(order):
        xs[i] = x_ref[i] * half_width + center
        ws[i] = w_ref[i] * half_width
        temp = q_single(xs[i], mu_array[0], sigma_array[0], ub_prev, b1_array[0], lb_prev, b2_array[0], node_array[1], x0, trunc_num, threshold, True)
        xs_prev[i] = xs[i]
        if log_space:
            if ws[i] * temp > 0:
                log_ws_pv_prev[i] = log(ws[i] * temp)
            else:
                log_ws_pv_prev[i] = NEG_INF
        else:
            ws_pv_prev[i] = ws[i] * temp

    ub_prev = ub
    lb_prev = lb

    # --- Intermediate stages ---
    for n in range(2, d):
        T_curr = node_array[n] - node_array[n - 1]
        ub = ub_prev + b1_array[n - 1] * T_curr
        lb = lb_prev + b2_array[n - 1] * T_curr
        half_width = (ub - lb) / 2.0
        center = (ub + lb) / 2.0

        for i in range(order):
            xs[i] = x_ref[i] * half_width + center
            ws[i] = w_ref[i] * half_width

            if log_space:
                for j in range(order):
                    temp = q_single(xs[i], mu_array[n - 1], sigma_array[n - 1], ub_prev, b1_array[n - 1], lb_prev, b2_array[n - 1], T_curr, xs_prev[j], trunc_num, threshold, True)
                    if temp > 0:
                        log_terms[j] = log(temp) + log_ws_pv_prev[j]
                    else:
                        log_terms[j] = NEG_INF
                temp = _logsumexp(log_terms, order)
                if ws[i] > 0:
                    log_ws_pv_new[i] = log(ws[i]) + temp
                else:
                    log_ws_pv_new[i] = NEG_INF
            else:
                pv[i] = 0
                for j in range(order):
                    temp = q_single(xs[i], mu_array[n - 1], sigma_array[n - 1], ub_prev, b1_array[n - 1], lb_prev, b2_array[n - 1], T_curr, xs_prev[j], trunc_num, threshold, True)
                    pv[i] += temp * ws_pv_prev[j]

        for i in range(order):
            xs_prev[i] = xs[i]
            if log_space:
                log_ws_pv_prev[i] = log_ws_pv_new[i]
            else:
                ws_pv_prev[i] = ws[i] * pv[i]

        ub_prev = ub
        lb_prev = lb

    # --- Final stage ---
    if log_space:
        for i in range(order):
            fptd_val = fptd_single(t - node_array[d - 1], mu_array[d - 1], sigma_array[d - 1], ub_prev, b1_array[d - 1], lb_prev, b2_array[d - 1], xs[i], bdy, trunc_num, threshold, True)
            if fptd_val > 0:
                log_terms[i] = log(fptd_val) + log_ws_pv_prev[i]
            else:
                log_terms[i] = NEG_INF
        return exp(_logsumexp(log_terms, order))
    else:
        for i in range(order):
            fptd_val = fptd_single(t - node_array[d - 1], mu_array[d - 1], sigma_array[d - 1], ub_prev, b1_array[d - 1], lb_prev, b2_array[d - 1], xs[i], bdy, trunc_num, threshold, True)
            result += fptd_val * ws_pv_prev[i]
        return result


# ---------------------------------------------------------------------------
# Public generalized multi-stage FPTD
# ---------------------------------------------------------------------------

cpdef double compute_heterog_multistage_fptd(
    double rt, int choice, double x0,
    double a1, double a2,
    double[:] mu_array, double[:] node_array,
    double[:] sigma_array,
    double[:] b1_array, double[:] b2_array,
    int d,
    int order=DEFAULT_QUADRATURE_ORDER, int trunc_num=DEFAULT_TRUNC_NUM,
    double threshold=DEFAULT_THRESHOLD,
    bint log_space=False,
):
    """Compute multi-stage FPTD with per-stage sigma and boundary slopes.

    Parameters
    ----------
    rt : double
        Time at which to evaluate the FPTD.
    choice : int
        +1 for upper boundary, -1 for lower boundary.
    x0 : double
        Starting position.
    a1 : double
        Upper boundary intercept at t=0.
    a2 : double
        Lower boundary intercept at t=0.
    mu_array : memoryview (d,)
        Drift rate per stage.
    node_array : memoryview (d,)
        Stage onset times (node_array[0] should be 0).
    sigma_array : memoryview (d,)
        Diffusion coefficient per stage.
    b1_array : memoryview (d,)
        Upper boundary slope per stage.
    b2_array : memoryview (d,)
        Lower boundary slope per stage.
    d : int
        Number of valid stages.
    order : int
        Gauss-Legendre quadrature order (default 30).
    trunc_num : int
        Series truncation limit.
    log_space : bool
        If True, use log-space computation to prevent underflow in deep models.
    """
    cdef double[:] x_ref
    cdef double[:] w_ref
    x_np, w_np = _get_quad_data(order)
    x_ref = x_np
    w_ref = w_np
    return _compute_multistage_fptd_core(
        rt, d, mu_array, node_array, sigma_array, a1, b1_array, a2, b2_array,
        x0, choice, order, x_ref, w_ref, trunc_num, threshold, log_space,
    )
