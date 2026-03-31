# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython-accelerated multi-stage log-FPTD computation.

Provides ``compute_addm_logfptd`` and
``compute_heterog_multistage_logfptd`` with split quadrature controls
(``order_mid`` for intermediate `q_single` propagation and ``order_last``
for the final `fptd_single` reduction). Batch log-likelihood helpers live in
``batch.pyx``.
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, M_PI, fabs, log, INFINITY
from libc.stdio cimport fprintf, stderr
from cython cimport boundscheck, wraparound, cdivision
from .utils cimport positive_log

from .single_stage cimport fptd_single, q_single
from ..quadrature import lgwt_lookup_table
from ..utils import resolve_quadrature_orders

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


cdef inline double _finish_addm_last_stage(
    double t,
    int d,
    double[:] mu_array,
    double[:] sacc_array,
    double sigma,
    double b,
    double a_final,
    int bdy,
    double* xs_last,
    double* ws_pv_last,
    double* log_ws_pv_last,
    double* log_terms,
    int order_last,
    int trunc_num,
    double threshold,
    bint log_space,
) noexcept nogil:
    cdef:
        int i
        double result = 0.0
        double fptd_val
        double t_in_final_stage = t - sacc_array[d - 1]

    if log_space:
        for i in range(order_last):
            fptd_val = fptd_single(
                t_in_final_stage,
                mu_array[d - 1],
                sigma,
                a_final,
                -b,
                -a_final,
                b,
                xs_last[i],
                bdy,
                trunc_num,
                threshold,
                True,
            )
            if fptd_val > 0:
                log_terms[i] = log(fptd_val) + log_ws_pv_last[i]
            else:
                log_terms[i] = -INFINITY
        return _logsumexp(log_terms, order_last)

    for i in range(order_last):
        fptd_val = fptd_single(
            t_in_final_stage,
            mu_array[d - 1],
            sigma,
            a_final,
            -b,
            -a_final,
            b,
            xs_last[i],
            bdy,
            trunc_num,
            threshold,
            True,
        )
        result += fptd_val * ws_pv_last[i]
    return positive_log(result)


cdef inline double _finish_general_last_stage(
    double t,
    int d,
    double[:] mu_array,
    double[:] node_array,
    double[:] sigma_array,
    double[:] b1_array,
    double[:] b2_array,
    double ub_final,
    double lb_final,
    int bdy,
    double* xs_last,
    double* ws_pv_last,
    double* log_ws_pv_last,
    double* log_terms,
    int order_last,
    int trunc_num,
    double threshold,
    bint log_space,
) noexcept nogil:
    cdef:
        int i
        double result = 0.0
        double fptd_val
        double t_in_final_stage = t - node_array[d - 1]

    if log_space:
        for i in range(order_last):
            fptd_val = fptd_single(
                t_in_final_stage,
                mu_array[d - 1],
                sigma_array[d - 1],
                ub_final,
                b1_array[d - 1],
                lb_final,
                b2_array[d - 1],
                xs_last[i],
                bdy,
                trunc_num,
                threshold,
                True,
            )
            if fptd_val > 0:
                log_terms[i] = log(fptd_val) + log_ws_pv_last[i]
            else:
                log_terms[i] = -INFINITY
        return _logsumexp(log_terms, order_last)

    for i in range(order_last):
        fptd_val = fptd_single(
            t_in_final_stage,
            mu_array[d - 1],
            sigma_array[d - 1],
            ub_final,
            b1_array[d - 1],
            lb_final,
            b2_array[d - 1],
            xs_last[i],
            bdy,
            trunc_num,
            threshold,
            True,
        )
        result += fptd_val * ws_pv_last[i]
    return positive_log(result)


# ---------------------------------------------------------------------------
# Core ADDM FPTD (unified normal / log-space)
# ---------------------------------------------------------------------------

cdef double _compute_addm_logfptd_core(
    double t, int d,
    double[:] mu_array, double[:] sacc_array,
    double sigma, double a, double b, double x0, int bdy,
    int order_mid, double[:] x_ref_mid_in, double[:] w_ref_mid_in,
    int order_last, double[:] x_ref_last_in, double[:] w_ref_last_in,
    int trunc_num, double threshold,
    bint log_space,
) noexcept nogil:
    """GIL-free core: compute ADDM log-FPTD in normal-space or log-space.

    When *log_space* is True, tracks log(ws*pv) instead of ws*pv to
    prevent underflow in deep (many-stage) models.
    """
    cdef:
        double result = 0.0
        int i, j, n
        double temp, fptd_val, a_curr, T_curr
        double NEG_INF = -INFINITY

        # Direct pointers into the memoryview data (no copy needed)
        double* x_ref_mid = &x_ref_mid_in[0]
        double* w_ref_mid = &w_ref_mid_in[0]
        double* x_ref_last = &x_ref_last_in[0]
        double* w_ref_last = &w_ref_last_in[0]

        # Stack buffers
        double xs_mid[MAX_ORDER]
        double ws_mid[MAX_ORDER]
        double xs_prev_mid[MAX_ORDER]
        double xs_last[MAX_ORDER]
        double ws_last[MAX_ORDER]
        # Normal-space buffers
        double pv_mid[MAX_ORDER]
        double ws_pv_prev_mid[MAX_ORDER]
        double pv_last[MAX_ORDER]
        double ws_pv_last[MAX_ORDER]
        # Log-space buffers
        double log_ws_pv_prev_mid[MAX_ORDER]
        double log_ws_pv_new_mid[MAX_ORDER]
        double log_ws_pv_last[MAX_ORDER]
        double log_terms[MAX_ORDER]

    if order_mid > MAX_ORDER:
        fprintf(stderr, "_compute_addm_logfptd_core: order_mid=%d exceeds MAX_ORDER=%d, clamping\n", order_mid, MAX_ORDER)
        order_mid = MAX_ORDER
    if order_last > MAX_ORDER:
        fprintf(stderr, "_compute_addm_logfptd_core: order_last=%d exceeds MAX_ORDER=%d, clamping\n", order_last, MAX_ORDER)
        order_last = MAX_ORDER

    if d == 1:
        return positive_log(
            fptd_single(
                t,
                mu_array[0],
                sigma,
                a,
                -b,
                -a,
                b,
                x0,
                bdy,
                trunc_num,
                threshold,
                True,
            )
        )

    if d == 2:
        for i in range(order_last):
            xs_last[i] = x_ref_last[i] * (a - b * sacc_array[1])
            ws_last[i] = w_ref_last[i] * (a - b * sacc_array[1])
            temp = q_single(xs_last[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold, True)
            if log_space:
                if ws_last[i] * temp > 0.0:
                    log_ws_pv_last[i] = log(ws_last[i] * temp)
                else:
                    log_ws_pv_last[i] = NEG_INF
            else:
                ws_pv_last[i] = ws_last[i] * temp

        a_curr = a - b * sacc_array[1]
        return _finish_addm_last_stage(
            t,
            d,
            mu_array,
            sacc_array,
            sigma,
            b,
            a_curr,
            bdy,
            xs_last,
            ws_pv_last,
            log_ws_pv_last,
            log_terms,
            order_last,
            trunc_num,
            threshold,
            log_space,
        )

    # --- First stage on the mid grid ---
    for i in range(order_mid):
        xs_mid[i] = x_ref_mid[i] * (a - b * sacc_array[1])
        ws_mid[i] = w_ref_mid[i] * (a - b * sacc_array[1])
        temp = q_single(xs_mid[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold, True)
        xs_prev_mid[i] = xs_mid[i]
        if log_space:
            if ws_mid[i] * temp > 0:
                log_ws_pv_prev_mid[i] = log(ws_mid[i] * temp)
            else:
                log_ws_pv_prev_mid[i] = NEG_INF
        else:
            ws_pv_prev_mid[i] = ws_mid[i] * temp

    # --- Mid-to-mid intermediate stages through stage d-3 ---
    for n in range(2, d - 1):
        for i in range(order_mid):
            xs_mid[i] = x_ref_mid[i] * (a - b * sacc_array[n])
            ws_mid[i] = w_ref_mid[i] * (a - b * sacc_array[n])
            a_curr = a - b * sacc_array[n - 1]
            T_curr = sacc_array[n] - sacc_array[n - 1]

            if log_space:
                for j in range(order_mid):
                    temp = q_single(xs_mid[i], mu_array[n - 1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev_mid[j], trunc_num, threshold, True)
                    if temp > 0:
                        log_terms[j] = log(temp) + log_ws_pv_prev_mid[j]
                    else:
                        log_terms[j] = NEG_INF
                temp = _logsumexp(log_terms, order_mid)
                if ws_mid[i] > 0:
                    log_ws_pv_new_mid[i] = log(ws_mid[i]) + temp
                else:
                    log_ws_pv_new_mid[i] = NEG_INF
            else:
                pv_mid[i] = 0
                for j in range(order_mid):
                    temp = q_single(xs_mid[i], mu_array[n - 1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev_mid[j], trunc_num, threshold, True)
                    pv_mid[i] += temp * ws_pv_prev_mid[j]

        for i in range(order_mid):
            xs_prev_mid[i] = xs_mid[i]
            if log_space:
                log_ws_pv_prev_mid[i] = log_ws_pv_new_mid[i]
            else:
                ws_pv_prev_mid[i] = ws_mid[i] * pv_mid[i]

    # --- Last q_single transition from the mid grid to the final-stage last grid ---
    a_curr = a - b * sacc_array[d - 1]
    T_curr = sacc_array[d - 1] - sacc_array[d - 2]
    for i in range(order_last):
        xs_last[i] = x_ref_last[i] * a_curr
        ws_last[i] = w_ref_last[i] * a_curr

        if log_space:
            for j in range(order_mid):
                temp = q_single(xs_last[i], mu_array[d - 2], sigma, a - b * sacc_array[d - 2], -b, -a + b * sacc_array[d - 2], b, T_curr, xs_prev_mid[j], trunc_num, threshold, True)
                if temp > 0:
                    log_terms[j] = log(temp) + log_ws_pv_prev_mid[j]
                else:
                    log_terms[j] = NEG_INF
            temp = _logsumexp(log_terms, order_mid)
            if ws_last[i] > 0:
                log_ws_pv_last[i] = log(ws_last[i]) + temp
            else:
                log_ws_pv_last[i] = NEG_INF
        else:
            pv_last[i] = 0
            for j in range(order_mid):
                temp = q_single(xs_last[i], mu_array[d - 2], sigma, a - b * sacc_array[d - 2], -b, -a + b * sacc_array[d - 2], b, T_curr, xs_prev_mid[j], trunc_num, threshold, True)
                pv_last[i] += temp * ws_pv_prev_mid[j]
            ws_pv_last[i] = ws_last[i] * pv_last[i]

    return _finish_addm_last_stage(
        t,
        d,
        mu_array,
        sacc_array,
        sigma,
        b,
        a_curr,
        bdy,
        xs_last,
        ws_pv_last,
        log_ws_pv_last,
        log_terms,
        order_last,
        trunc_num,
        threshold,
        log_space,
    )


# ---------------------------------------------------------------------------
# Public ADDM log-FPTD
# ---------------------------------------------------------------------------

cpdef double compute_addm_logfptd(
    double rt, int choice,
    double eta, double kappa, double sigma, double a, double b, double x0,
    double r1, double r2, int flag,
    double[:] sacc_array, int d,
    int order_mid=DEFAULT_MID_QUAD_ORDER,
    int order_last=DEFAULT_LAST_QUAD_ORDER,
    object order=None,
    int trunc_num=DEFAULT_TRUNC_NUM,
    double threshold=DEFAULT_THRESHOLD,
    bint log_space=False,
):
    """Compute the ADDM log-likelihood of hitting *choice* at time *rt*.

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
    order_mid : int
        Intermediate-stage Gauss-Legendre quadrature order (default 20).
    order_last : int
        Final-stage Gauss-Legendre quadrature order (default 30).
    order : int or None
        Legacy compatibility alias. If provided on its own, it maps to both
        split quadrature orders.
    trunc_num : int
        Series truncation limit.
    log_space : bool
        If True, use log-space accumulation internally before returning the
        log-density.
    """
    cdef:
        double[:] x_ref_mid
        double[:] w_ref_mid
        double[:] x_ref_last
        double[:] w_ref_last
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

    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    x_mid_np, w_mid_np = _get_quad_data(order_mid)
    x_last_np, w_last_np = _get_quad_data(order_last)
    x_ref_mid = x_mid_np
    w_ref_mid = w_mid_np
    x_ref_last = x_last_np
    w_ref_last = w_last_np

    return _compute_addm_logfptd_core(
        rt, d, mu_array, sacc_array, sigma, a, b, x0, choice,
        order_mid, x_ref_mid, w_ref_mid, order_last, x_ref_last, w_ref_last,
        trunc_num, threshold, log_space,
    )


# ---------------------------------------------------------------------------
# Core generalized multi-stage FPTD (unified normal / log-space)
# ---------------------------------------------------------------------------

cdef double _compute_multistage_logfptd_core(
    double t, int d,
    double[:] mu_array, double[:] node_array,
    double[:] sigma_array,
    double a1, double[:] b1_array,
    double a2, double[:] b2_array,
    double x0, int bdy,
    int order_mid, double[:] x_ref_mid_in, double[:] w_ref_mid_in,
    int order_last, double[:] x_ref_last_in, double[:] w_ref_last_in,
    int trunc_num, double threshold,
    bint log_space,
) noexcept nogil:
    """GIL-free core: generalized multi-stage log-FPTD with per-stage sigma and slopes."""
    cdef:
        double result = 0.0
        int i, j, n, last_q_stage_idx
        double temp, fptd_val, T_curr
        double ub, lb, ub_prev, lb_prev, ub_final, lb_final
        double half_width, center
        double NEG_INF = -INFINITY

        # Direct pointers into the memoryview data (no copy needed)
        double* x_ref_mid = &x_ref_mid_in[0]
        double* w_ref_mid = &w_ref_mid_in[0]
        double* x_ref_last = &x_ref_last_in[0]
        double* w_ref_last = &w_ref_last_in[0]

        # Stack buffers
        double xs_mid[MAX_ORDER]
        double ws_mid[MAX_ORDER]
        double xs_prev_mid[MAX_ORDER]
        double xs_last[MAX_ORDER]
        double ws_last[MAX_ORDER]
        # Normal-space buffers
        double pv_mid[MAX_ORDER]
        double ws_pv_prev_mid[MAX_ORDER]
        double pv_last[MAX_ORDER]
        double ws_pv_last[MAX_ORDER]
        # Log-space buffers
        double log_ws_pv_prev_mid[MAX_ORDER]
        double log_ws_pv_new_mid[MAX_ORDER]
        double log_ws_pv_last[MAX_ORDER]
        double log_terms[MAX_ORDER]

    if order_mid > MAX_ORDER:
        fprintf(
            stderr,
            "_compute_multistage_logfptd_core: order_mid=%d exceeds MAX_ORDER=%d, clamping\n",
            order_mid,
            MAX_ORDER,
        )
        order_mid = MAX_ORDER
    if order_last > MAX_ORDER:
        fprintf(
            stderr,
            "_compute_multistage_logfptd_core: order_last=%d exceeds MAX_ORDER=%d, clamping\n",
            order_last,
            MAX_ORDER,
        )
        order_last = MAX_ORDER

    if d == 1:
        return positive_log(
            fptd_single(
                t,
                mu_array[0],
                sigma_array[0],
                a1,
                b1_array[0],
                a2,
                b2_array[0],
                x0,
                bdy,
                trunc_num,
                threshold,
                True,
            )
        )

    if d == 2:
        T_curr = node_array[1] - node_array[0]
        ub_final = a1 + b1_array[0] * T_curr
        lb_final = a2 + b2_array[0] * T_curr
        half_width = (ub_final - lb_final) / 2.0
        center = (ub_final + lb_final) / 2.0

        for i in range(order_last):
            xs_last[i] = x_ref_last[i] * half_width + center
            ws_last[i] = w_ref_last[i] * half_width
            temp = q_single(
                xs_last[i],
                mu_array[0],
                sigma_array[0],
                a1,
                b1_array[0],
                a2,
                b2_array[0],
                T_curr,
                x0,
                trunc_num,
                threshold,
                True,
            )
            if log_space:
                log_ws_pv_last[i] = positive_log(ws_last[i] * temp)
            else:
                ws_pv_last[i] = ws_last[i] * temp

        return _finish_general_last_stage(
            t,
            d,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            ub_final,
            lb_final,
            bdy,
            xs_last,
            ws_pv_last,
            log_ws_pv_last,
            log_terms,
            order_last,
            trunc_num,
            threshold,
            log_space,
        )

    # First q propagation onto the first intermediate-stage grid.
    T_curr = node_array[1] - node_array[0]
    ub = a1 + b1_array[0] * T_curr
    lb = a2 + b2_array[0] * T_curr
    half_width = (ub - lb) / 2.0
    center = (ub + lb) / 2.0

    for i in range(order_mid):
        xs_mid[i] = x_ref_mid[i] * half_width + center
        ws_mid[i] = w_ref_mid[i] * half_width
        temp = q_single(
            xs_mid[i],
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            T_curr,
            x0,
            trunc_num,
            threshold,
            True,
        )
        xs_prev_mid[i] = xs_mid[i]
        if log_space:
            log_ws_pv_prev_mid[i] = positive_log(ws_mid[i] * temp)
        else:
            ws_pv_prev_mid[i] = ws_mid[i] * temp

    ub_prev = ub
    lb_prev = lb

    # Intermediate q-only transitions up to the second-to-last stage.
    for n in range(2, d - 1):
        T_curr = node_array[n] - node_array[n - 1]
        ub = ub_prev + b1_array[n - 1] * T_curr
        lb = lb_prev + b2_array[n - 1] * T_curr
        half_width = (ub - lb) / 2.0
        center = (ub + lb) / 2.0

        for i in range(order_mid):
            xs_mid[i] = x_ref_mid[i] * half_width + center
            ws_mid[i] = w_ref_mid[i] * half_width

            if log_space:
                for j in range(order_mid):
                    temp = q_single(
                        xs_mid[i],
                        mu_array[n - 1],
                        sigma_array[n - 1],
                        ub_prev,
                        b1_array[n - 1],
                        lb_prev,
                        b2_array[n - 1],
                        T_curr,
                        xs_prev_mid[j],
                        trunc_num,
                        threshold,
                        True,
                    )
                    if temp > 0:
                        log_terms[j] = log(temp) + log_ws_pv_prev_mid[j]
                    else:
                        log_terms[j] = NEG_INF
                temp = _logsumexp(log_terms, order_mid)
                if ws_mid[i] > 0:
                    log_ws_pv_new_mid[i] = log(ws_mid[i]) + temp
                else:
                    log_ws_pv_new_mid[i] = NEG_INF
            else:
                pv_mid[i] = 0
                for j in range(order_mid):
                    temp = q_single(
                        xs_mid[i],
                        mu_array[n - 1],
                        sigma_array[n - 1],
                        ub_prev,
                        b1_array[n - 1],
                        lb_prev,
                        b2_array[n - 1],
                        T_curr,
                        xs_prev_mid[j],
                        trunc_num,
                        threshold,
                        True,
                    )
                    pv_mid[i] += temp * ws_pv_prev_mid[j]

        for i in range(order_mid):
            xs_prev_mid[i] = xs_mid[i]
            if log_space:
                log_ws_pv_prev_mid[i] = log_ws_pv_new_mid[i]
            else:
                ws_pv_prev_mid[i] = ws_mid[i] * pv_mid[i]

        ub_prev = ub
        lb_prev = lb

    # Final q transition onto the last-stage grid.
    last_q_stage_idx = d - 2
    T_curr = node_array[d - 1] - node_array[d - 2]
    ub_final = ub_prev + b1_array[last_q_stage_idx] * T_curr
    lb_final = lb_prev + b2_array[last_q_stage_idx] * T_curr
    half_width = (ub_final - lb_final) / 2.0
    center = (ub_final + lb_final) / 2.0

    for i in range(order_last):
        xs_last[i] = x_ref_last[i] * half_width + center
        ws_last[i] = w_ref_last[i] * half_width
        if log_space:
            for j in range(order_mid):
                temp = q_single(
                    xs_last[i],
                    mu_array[last_q_stage_idx],
                    sigma_array[last_q_stage_idx],
                    ub_prev,
                    b1_array[last_q_stage_idx],
                    lb_prev,
                    b2_array[last_q_stage_idx],
                    T_curr,
                    xs_prev_mid[j],
                    trunc_num,
                    threshold,
                    True,
                )
                if temp > 0:
                    log_terms[j] = log(temp) + log_ws_pv_prev_mid[j]
                else:
                    log_terms[j] = NEG_INF
            temp = _logsumexp(log_terms, order_mid)
            if ws_last[i] > 0:
                log_ws_pv_last[i] = log(ws_last[i]) + temp
            else:
                log_ws_pv_last[i] = NEG_INF
        else:
            pv_last[i] = 0
            for j in range(order_mid):
                temp = q_single(
                    xs_last[i],
                    mu_array[last_q_stage_idx],
                    sigma_array[last_q_stage_idx],
                    ub_prev,
                    b1_array[last_q_stage_idx],
                    lb_prev,
                    b2_array[last_q_stage_idx],
                    T_curr,
                    xs_prev_mid[j],
                    trunc_num,
                    threshold,
                    True,
                )
                pv_last[i] += temp * ws_pv_prev_mid[j]
            ws_pv_last[i] = ws_last[i] * pv_last[i]

    return _finish_general_last_stage(
        t,
        d,
        mu_array,
        node_array,
        sigma_array,
        b1_array,
        b2_array,
        ub_final,
        lb_final,
        bdy,
        xs_last,
        ws_pv_last,
        log_ws_pv_last,
        log_terms,
        order_last,
        trunc_num,
        threshold,
        log_space,
    )


# ---------------------------------------------------------------------------
# Public generalized multi-stage log-FPTD
# ---------------------------------------------------------------------------

cpdef double compute_heterog_multistage_logfptd(
    double rt, int choice, double x0,
    double a1, double a2,
    double[:] mu_array, double[:] node_array,
    double[:] sigma_array,
    double[:] b1_array, double[:] b2_array,
    int d,
    int order_mid=DEFAULT_MID_QUAD_ORDER,
    int order_last=DEFAULT_LAST_QUAD_ORDER,
    object order=None,
    int trunc_num=DEFAULT_TRUNC_NUM,
    double threshold=DEFAULT_THRESHOLD,
    bint log_space=False,
):
    """Compute multi-stage log-FPTD with per-stage sigma and boundary slopes.

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
    order_mid : int
        Intermediate-stage Gauss-Legendre quadrature order (default 20).
    order_last : int
        Final-stage Gauss-Legendre quadrature order (default 30).
    order : int or None
        Legacy compatibility alias that maps to both split orders.
    trunc_num : int
        Series truncation limit.
    log_space : bool
        If True, use log-space accumulation internally before returning the
        log-density.
    """
    cdef double[:] x_ref_mid
    cdef double[:] w_ref_mid
    cdef double[:] x_ref_last
    cdef double[:] w_ref_last

    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    x_mid_np, w_mid_np = _get_quad_data(order_mid)
    x_last_np, w_last_np = _get_quad_data(order_last)
    x_ref_mid = x_mid_np
    w_ref_mid = w_mid_np
    x_ref_last = x_last_np
    w_ref_last = w_last_np
    return _compute_multistage_logfptd_core(
        rt, d, mu_array, node_array, sigma_array, a1, b1_array, a2, b2_array,
        x0, choice,
        order_mid, x_ref_mid, w_ref_mid,
        order_last, x_ref_last, w_ref_last,
        trunc_num, threshold, log_space,
    )
