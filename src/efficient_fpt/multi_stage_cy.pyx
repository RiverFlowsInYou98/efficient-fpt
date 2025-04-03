# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, M_PI, fabs, log
from cython cimport boundscheck, wraparound, cdivision, language_level
from cython.parallel import prange
import time

from .single_stage_cy cimport fptd_single_cy, q_single_cy

cdef extern from "omp.h":
    int omp_get_num_threads()
    int omp_get_max_threads()

cpdef double get_addm_fptd_cy(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order = 20
        double x_ref20[20]
        double w_ref20[20]
        double xs[20]
        double ws[20]
        double xs_prev[20]
        double pv[20]
        double ws_pv_product_prev[20]
        double temp
        double fptd
        double a_curr, T_curr

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref20[:] = [-0.9931285991850949, -0.9639719272779138, -0.9122344282513258, -0.8391169718222188, -0.7463319064601508,
                          -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.2277858511416451, -0.07652652113349734,
                          0.07652652113349734, 0.2277858511416451, 0.37370608871541955, 0.5108670019508271, 0.636053680726515,
                          0.7463319064601508, 0.8391169718222188, 0.9122344282513258, 0.9639719272779138, 0.9931285991850949]
        w_ref20[:] = [0.017614007139153273, 0.04060142980038622, 0.06267204833410944, 0.08327674157670467, 0.10193011981724026,
                          0.11819453196151825, 0.13168863844917653, 0.14209610931838187, 0.14917298647260366, 0.15275338713072578,
                          0.15275338713072578, 0.14917298647260366, 0.14209610931838187, 0.13168863844917653, 0.11819453196151825,
                          0.10193011981724026, 0.08327674157670467, 0.06267204833410944, 0.04060142980038622, 0.017614007139153273]
        for i in range(order):
            xs[i] = x_ref20[i] * (a - b * sacc_array[1])
            ws[i] = w_ref20[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref20[i] * (a - b * sacc_array[n])
                ws[i] = w_ref20[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef run_timings():
    cdef:
        double sigma = 1.0
        double a = 1.5
        double b = 0.3
        double x0 = -0.5
        int d
        np.ndarray[np.double_t, ndim=1] mu_array
        np.ndarray[np.double_t, ndim=1] fixation_array
        np.ndarray[np.double_t, ndim=1] sacc_array
        np.ndarray[np.double_t, ndim=1] rt_array

    mu_array = np.array([1., -0.2, 1.5, 0.5, -1., 1., -0.2, 1.5, 0.5, -1.], dtype=np.float64)
    fixation_array = np.array([0.5, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.25, 0.5], dtype=np.float64)
    sacc_array = np.cumsum(fixation_array).astype(np.float64)
    sacc_array = np.concatenate(([0.0], sacc_array)).astype(np.float64)
    rt_array = ((sacc_array[1:] + sacc_array[:-1]) / 2).astype(np.float64)
    sacc_array = sacc_array[:-1]
    d = mu_array.shape[0]

    # Run tests
    for n in range(d):
        time_get_addm_fptd(rt_array[n], n + 1, mu_array, sacc_array, sigma, a, b, x0, 1)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef time_get_addm_fptd(double t, int d, np.ndarray[double, ndim=1] mu_array, np.ndarray[double, ndim=1] sacc_array, double sigma, double a, double b, double x0, int bdy):
    cdef:
        int n_runs = 100
        int run
        double start, end, duration

    start = time.perf_counter()
    for run in range(n_runs):
        get_addm_fptd_cy(t, d, mu_array, sacc_array, sigma, a, b, x0, bdy)
    end = time.perf_counter()

    duration = (end - start) * 1e6
    print("%d stages, average time per run: %.2f microseconds" % (d, duration / n_runs))
    


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef get_mu_array_padded(double mu1, double mu2, int max_d, int d, int flag):
    """
    Generate an array of length `max_d`.
    The first `d` elements are alternating drift rates `mu1` and `mu2`, where `flag` determines the starting drift rate.
    `flag`=0: mu1 -> mu2 -> mu1 -> ...
    `flag`=1: mu2 -> mu1 -> mu2 -> ...
    The remaining elements are set to default(0). 
    Note that the effective length of mu_array is `d` and the effective length of sacc_array is `d-1`.
    """
    cdef np.ndarray[double, ndim=1] mu_array = np.zeros(max_d, dtype=np.float64)
    cdef double current_mu = mu2 if flag else mu1
    for i in range(d):
        mu_array[i] = current_mu
        current_mu = mu2 if current_mu == mu1 else mu1
    return mu_array

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef get_mu_data_padded(double mu1, double mu2, int max_d, np.ndarray[int, ndim=1] flag_data, np.ndarray[int, ndim=1] length_data):
    """
    Generate a dataset of drift rates
    each drift rate is from `get_mu_array_padded(mu1, mu2, max_d, length_data[i], flag_data[i])`
    """
    cdef int num_data = len(flag_data)
    cdef np.ndarray[double, ndim=2] mu_data = np.zeros((num_data, max_d), dtype=np.float64)
    for n in range(num_data):
        mu_data[n] = get_mu_array_padded(mu1, mu2, max_d, length_data[n], flag_data[n])
    return mu_data


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_loss_serial(double mu1, double mu2, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    cdef:
        int n
        double total_loss = 0.0, likelihood, loss
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_data=get_mu_data_padded(mu1, mu2, max_d, flag_data, length_data)
    for n in range(num_data):
        likelihood = get_addm_fptd_cy(rt_data[n], length_data[n], mu_data[n], sacc_data[n], sigma, a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            loss = 0
            num_data_effective -= 1
        # if n % 1 == 0:
        #     print(f"n={n}, -loglikelihood={loss:.5f}")
        total_loss += loss
    return total_loss / num_data_effective


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_loss_parallel(double mu1, double mu2, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    cdef:
        int n
        double total_loss = 0.0, likelihood, loss
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_data=get_mu_data_padded(mu1, mu2, max_d, flag_data, length_data)
        double[:, :] mu_data_view = mu_data
        double[:, :] sacc_data_view = sacc_data
    max_num_threads = omp_get_max_threads()
    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=max_num_threads):
        likelihood = get_addm_fptd_cy(rt_data[n], length_data[n], mu_data_view[n], sacc_data_view[n], sigma, a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            loss = 0
            num_data_effective -= 1
        with gil:
            total_loss += loss
    # print("num_data_effective:", num_data_effective)
    return total_loss / num_data_effective

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_glamloss_parallel(double mu1, double mu2, \
                                        np.ndarray[double, ndim=1] rt_data, \
                                        np.ndarray[int, ndim=1] choice_data, \
                                        np.ndarray[int, ndim=1] flag_data, \
                                        np.ndarray[double, ndim=2] sacc_data, \
                                        np.ndarray[int, ndim=1] length_data, \
                                        int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    cdef:
        int n
        double total_loss = 0.0, likelihood, loss
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_data=get_mu_data_padded(mu1, mu2, max_d, flag_data, length_data)
        double[:, :] mu_data_view = mu_data
        double[:, :] sacc_data_view = sacc_data
        double[:] approx_mu_data = np.zeros(num_data)
    for n in range(num_data):
        approx_mu_data[n] = mu_data[n, 0] * sacc_data[n, 0] + mu_data[n, length_data[n] - 1] * (rt_data[n] - sacc_data[n, length_data[n] - 2])
        for i in range(1, length_data[n] - 1):
            approx_mu_data[n] += mu_data[n, i] * (sacc_data[n, i] - sacc_data[n, i - 1])
        approx_mu_data[n] /= rt_data[n]
    max_num_threads = omp_get_max_threads()
    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=max_num_threads):
        likelihood = fptd_single_cy(rt_data[n], approx_mu_data[n], sigma, a, -b, -a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            loss = 0
            num_data_effective -= 1
        with gil:
            total_loss += loss
    return total_loss / num_data_effective

cpdef print_num_threads():
    print("Number of available threads:", omp_get_max_threads())


