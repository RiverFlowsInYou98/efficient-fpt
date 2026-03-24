# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython-accelerated multi-stage FPT density computation.

Provides ``get_addm_fptd_cy`` (parameterized by quadrature *order*) and
batch likelihood helpers ``compute_llhds_serial``, ``compute_loss_serial``,
``compute_loss_parallel``.
"""

import warnings
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, M_PI, fabs, log
from libc.stdio cimport fprintf, stderr
from cython cimport boundscheck, wraparound, cdivision
from cython.parallel import prange

from .single_stage_cy cimport fptd_single_cy, q_single_cy
from .utils import build_mu_array_data

cdef extern from "omp.h":
    int omp_get_num_threads()
    int omp_get_max_threads()

# ---------------------------------------------------------------------------
# Quadrature data cache (single source of truth)
# ---------------------------------------------------------------------------
# Pre-populated with commonly used orders at module load time.
# Any order can be requested; missing ones are computed on the fly via leggauss.

_QUAD_CACHE = {}

def _get_quad_data(int order):
    """Return (x_ref, w_ref) arrays on [-1, 1] for the given order."""
    if order not in _QUAD_CACHE:
        x, w = np.polynomial.legendre.leggauss(order)
        _QUAD_CACHE[order] = (
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64),
        )
    return _QUAD_CACHE[order]

# Pre-populate common orders
for _order in [10, 15, 20, 25, 30, 35, 40, 100]:
    _get_quad_data(_order)

# Maximum quadrature order supported by the stack buffers.
DEF MAX_ORDER = 200


# ---------------------------------------------------------------------------
# Core nogil FPTD computation
# ---------------------------------------------------------------------------

cdef double _get_addm_fptd_impl(
    double t, int d,
    double[:] mu_array, double[:] sacc_array,
    double sigma, double a, double b, double x0, int bdy,
    int order, double[:] x_ref_in, double[:] w_ref_in,
    int trunc_num, double threshold,
) nogil:
    """GIL-free core: compute the likelihood of hitting boundary *bdy* at time *t*.

    Quadrature nodes ``x_ref_in`` and weights ``w_ref_in`` (on [-1, 1]) must
    be pre-looked-up by the caller while the GIL is held.
    """
    cdef:
        double result = 0.0
        int i, j, n
        double temp, fptd_val, a_curr, T_curr

        # Stack buffers – sized to MAX_ORDER (compile-time constant).
        double x_ref[MAX_ORDER]
        double w_ref[MAX_ORDER]
        double xs[MAX_ORDER]
        double ws[MAX_ORDER]
        double xs_prev[MAX_ORDER]
        double pv[MAX_ORDER]
        double ws_pv_product_prev[MAX_ORDER]

    if order > MAX_ORDER:
        fprintf(stderr, "_get_addm_fptd_impl: order=%d exceeds MAX_ORDER=%d, clamping\n", order, MAX_ORDER)
        order = MAX_ORDER

    if d == 1:
        return fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)

    # Copy reference nodes/weights into stack arrays
    for i in range(order):
        x_ref[i] = x_ref_in[i]
        w_ref[i] = w_ref_in[i]

    # --- First stage ---
    for i in range(order):
        xs[i] = x_ref[i] * (a - b * sacc_array[1])
        ws[i] = w_ref[i] * (a - b * sacc_array[1])
        pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
        xs_prev[i] = xs[i]
        ws_pv_product_prev[i] = ws[i] * pv[i]

    # --- Intermediate stages ---
    for n in range(2, d):
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[n])
            ws[i] = w_ref[i] * (a - b * sacc_array[n])
            pv[i] = 0
            a_curr = a - b * sacc_array[n - 1]
            T_curr = sacc_array[n] - sacc_array[n - 1]
            for j in range(order):
                temp = q_single_cy(xs[i], mu_array[n - 1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                pv[i] += temp * ws_pv_product_prev[j]
        for i in range(order):
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]

    # --- Final stage: accumulate FPTD ---
    a_curr = a - b * sacc_array[d - 1]
    for i in range(order):
        fptd_val = fptd_single_cy(t - sacc_array[d - 1], mu_array[d - 1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
        result += fptd_val * ws_pv_product_prev[i]

    return result


# ---------------------------------------------------------------------------
# Public FPTD function (Python-callable, looks up quadrature data)
# ---------------------------------------------------------------------------

cpdef double get_addm_fptd_cy(
    double t, int d,
    double[:] mu_array, double[:] sacc_array,
    double sigma, double a, double b, double x0, int bdy,
    int trunc_num=100, double threshold=1e-20, int order=30,
):
    """Compute the likelihood of hitting boundary *bdy* at time *t*.

    Parameters
    ----------
    t : double
        Time at which to evaluate the FPTD.
    d : int
        Number of stages.
    mu_array : memoryview (d,)
        Drift rate per stage.
    sacc_array : memoryview (d,)
        Stage onset times.
    sigma, a, b, x0 : double
        Diffusion coefficient, boundary intercept/slope, starting point.
    bdy : int
        +1 for upper boundary, -1 for lower boundary.
    trunc_num : int
        Series truncation limit.
    threshold : double
        Early-stopping threshold for series terms.
    order : int
        Gauss-Legendre quadrature order (default 30).
    """
    cdef double[:] x_ref
    cdef double[:] w_ref
    x_np, w_np = _get_quad_data(order)
    x_ref = x_np
    w_ref = w_np
    return _get_addm_fptd_impl(
        t, d, mu_array, sacc_array, sigma, a, b, x0, bdy,
        order, x_ref, w_ref, trunc_num, threshold,
    )


# ---------------------------------------------------------------------------
# Batch likelihood computation
# ---------------------------------------------------------------------------

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial(
    np.ndarray[double, ndim=1] mu1_data,
    np.ndarray[double, ndim=1] mu2_data,
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int max_d, double sigma, double a, double b, double x0,
    double threshold=1e-20, int order=30,
):
    """Per-trial likelihoods (serial). Returns 1-D array; 0.0 for invalid trials.

    Caller must pre-swap mu1/mu2 for trials where the opposite fixation order is needed.
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = build_mu_array_data(
            mu1_data, mu2_data, d_data, max_d,
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy(
            rt_data[n], d_data[n], mu_array_data[n], sacc_array_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold, order,
        )
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            likelihoods[n] = 0.0
    return likelihoods


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_loss_serial(
    np.ndarray[double, ndim=1] mu1_data,
    np.ndarray[double, ndim=1] mu2_data,
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int max_d, double sigma, double a, double b, double x0,
    double threshold=1e-20, int order=30,
):
    """Mean negative log-likelihood (serial)."""
    cdef:
        int n
        double total_loss = 0.0, likelihood, loss
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_array_data = build_mu_array_data(
            mu1_data, mu2_data, d_data, max_d,
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy(
            rt_data[n], d_data[n], mu_array_data[n], sacc_array_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold, order,
        )
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            loss = 0
            num_data_effective -= 1
        total_loss += loss
    if num_data_effective == 0:
        return float("nan")
    return total_loss / num_data_effective


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_loss_parallel(
    np.ndarray[double, ndim=1] mu1_data,
    np.ndarray[double, ndim=1] mu2_data,
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int max_d, double sigma, double a, double b, double x0,
    double threshold=1e-20, int num_threads=-1, int order=30,
):
    """Mean negative log-likelihood (OpenMP parallel).

    Caller must pre-swap mu1/mu2 for trials where the opposite fixation order is needed.
    """
    cdef:
        int n
        double total_loss = 0.0, likelihood
        int num_data = len(rt_data), num_data_effective = 0
        np.ndarray[double, ndim=2] mu_array_data = build_mu_array_data(
            mu1_data, mu2_data, d_data, max_d,
        )
        double[:, :] mu_array_data_view = mu_array_data
        double[:, :] sacc_array_data_view = sacc_array_data
        np.ndarray[double, ndim=1] loss_data = np.zeros(num_data, dtype=np.float64)
        np.ndarray[np.uint8_t, ndim=1] valid_data = np.zeros(num_data, dtype=np.uint8)
        double[:] loss_view = loss_data
        np.uint8_t[:] valid_view = valid_data

    # Look up quadrature data while GIL is held
    x_np, w_np = _get_quad_data(order)
    cdef double[:] x_ref = x_np
    cdef double[:] w_ref = w_np

    if num_threads <= 0:
        num_threads = omp_get_max_threads()

    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=num_threads):
        likelihood = _get_addm_fptd_impl(
            rt_data[n], d_data[n], mu_array_data_view[n], sacc_array_data_view[n],
            sigma, a, b, x0, choice_data[n],
            order, x_ref, w_ref, 100, threshold,
        )
        if likelihood > 0:
            loss_view[n] = -log(likelihood)
            valid_view[n] = 1
        else:
            loss_view[n] = 0.0
            valid_view[n] = 0
    for n in range(num_data):
        total_loss += loss_view[n]
        num_data_effective += valid_view[n]
    if num_data_effective == 0:
        return float("nan")
    return total_loss / num_data_effective


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_tadaloss_parallel(
    np.ndarray[double, ndim=1] mu1_data,
    np.ndarray[double, ndim=1] mu2_data,
    np.ndarray[double, ndim=1] rt_data,
    np.ndarray[int, ndim=1] choice_data,
    np.ndarray[double, ndim=2] sacc_array_data,
    np.ndarray[int, ndim=1] d_data,
    int max_d, double sigma, double a, double b, double x0,
    double threshold=1e-20,
):
    """Mean negative log-likelihood using time-averaged drift approximation (parallel)."""
    cdef:
        int n, i, L
        double total_loss = 0.0, likelihood, mu_sum
        int num_data = len(rt_data), num_data_effective = 0
        np.ndarray[double, ndim=2] mu_array_data = build_mu_array_data(
            mu1_data, mu2_data, d_data, max_d,
        )
        double[:, :] mu_array_data_view = mu_array_data
        double[:, :] sacc_array_data_view = sacc_array_data
        double[:] mu_tada_data = np.zeros(num_data)
        np.ndarray[double, ndim=1] loss_data = np.zeros(num_data, dtype=np.float64)
        np.ndarray[np.uint8_t, ndim=1] valid_data = np.zeros(num_data, dtype=np.uint8)
        double[:] loss_view = loss_data
        np.uint8_t[:] valid_view = valid_data
    cdef int max_num_threads

    for n in range(num_data):
        L = d_data[n]
        if rt_data[n] > 0.0:
            mu_sum = 0.0
            for i in range(L - 1):
                mu_sum += mu_array_data_view[n, i] * (sacc_array_data_view[n, i + 1] - sacc_array_data_view[n, i])
            mu_sum += mu_array_data_view[n, L - 1] * (rt_data[n] - sacc_array_data_view[n, L - 1])
            mu_tada_data[n] = mu_sum / rt_data[n]
        else:
            warnings.warn(f"compute_tadaloss_parallel: trial {n} has rt={rt_data[n]}, skipping")
            mu_tada_data[n] = 0.0

    max_num_threads = omp_get_max_threads()
    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=max_num_threads):
        likelihood = fptd_single_cy(rt_data[n], mu_tada_data[n], sigma, a, -b, -a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss_view[n] = -log(likelihood)
            valid_view[n] = 1
        else:
            loss_view[n] = 0.0
            valid_view[n] = 0
    for n in range(num_data):
        total_loss += loss_view[n]
        num_data_effective += valid_view[n]
    if num_data_effective == 0:
        return float("nan")
    return total_loss / num_data_effective


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

cpdef print_num_threads():
    """Print the number of available OpenMP threads."""
    print("Number of available threads:", omp_get_max_threads())
