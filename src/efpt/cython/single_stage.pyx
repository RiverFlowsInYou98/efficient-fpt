# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, M_PI, fabs
from libc.stdio cimport fprintf, stderr
from cython cimport boundscheck, wraparound, cdivision, language_level
from .utils cimport positive_log

include "_defaults.pxi"

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef inline double fptd_basic(double t, double mu, double a1, double b1, double a2, double b2, int bdy, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True) noexcept nogil:
    """
    First passage time density of Brownian motion with drift starting at x0 = 0 to
    the upper boundary u(t) = a1 + b1 * t and the lower boundary l(t) = a2 + b2 * t
    where a1 > 0 > a2, b1 < 0 < b2
    Note: set a1 = -a2 = a, b1 = b2 = -b to recover the symmetric case
    t shoud be in (0, -(a1 - a2) / (b1 - b2)), otherwise the density is 0
    """
    cdef double a_bar = (a1 + a2) / 2
    cdef double b = (b2 - b1) / 2
    cdef double c = a1 - a2
    cdef double delta, factor, result = 0, rj, term
    cdef int j
    if t <= 0.0:
        return 0.0
    if b2 > b1 and t >= c / (b2 - b1):
        return 0.0
    if bdy == 1:
        delta = mu - b1
        factor = pow(t, -1.5) / sqrt(2 * M_PI) * exp(-b / c * a1**2 + a1 * delta - 0.5 * delta**2 * t)
    elif bdy == -1:
        delta = -mu + b2
        factor = pow(t, -1.5) / sqrt(2 * M_PI) * exp(-b / c * a2**2 - a2 * delta - 0.5 * delta**2 * t)
    else:
        fprintf(stderr, "fptd_basic: invalid bdy=%d, returning 0.0\n", bdy)
        return 0.0
    cdef double sign = 1.0
    if adaptive_stopping:
        for j in range(trunc_num):
            rj = (j + 0.5) * c + bdy * sign * a_bar
            term = sign * rj * exp((b / c - 1 / (2 * t)) * rj**2)
            if fabs(term) < threshold:
                break
            result += term
            sign = -sign
    else:
        for j in range(trunc_num):
            rj = (j + 0.5) * c + bdy * sign * a_bar
            term = sign * rj * exp((b / c - 1 / (2 * t)) * rj**2)
            result += term
            sign = -sign
    return result * factor


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef inline double q_basic(double x, double mu, double a1, double b1, double a2, double b2, double T, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True) noexcept nogil:
    """
    density of Brownian motion with drift at time T starting at x0 = 0 
    given that it hasn't hit the upper boundary u(t) = a1 + b1 * t or the lower boundary l(t) = a2 + b2 * t
    upper boundary: u(t) = a1 + b1 * t
    lower boundary: l(t) = a2 + b2 * t
    vertical boundary: v(x) = T
    where a1 > 0 > a2, b1 < 0 < b2, T > 0
    Note: set a1 = -a2 = a, b1 = b2 = -b to recover the symmetric case
    x shoud be in (l(T), u(T)), otherwise the density is 0
    """
    cdef double a_bar = (a1 + a2) / 2
    cdef double b = (b2 - b1) / 2
    cdef double b_bar = (b1 + b2) / 2
    cdef double c = a1 - a2
    cdef double upper_T
    cdef double lower_T
    if T <= 0.0:
        return 0.0
    if b2 > b1 and T >= c / (b2 - b1):
        return 0.0
    upper_T = a1 + b1 * T
    lower_T = a2 + b2 * T
    if x <= lower_T or x >= upper_T:
        return 0.0
    cdef double y = x - b_bar * T
    cdef double factor = exp((mu - b_bar) * x - 0.5 * (mu**2 - b_bar**2) * T) / sqrt(T)
    cdef double result = 1 / sqrt(2 * M_PI) * exp(-(y**2) / (2 * T))
    cdef double t1, t2, t3, t4, term
    cdef int j
    if adaptive_stopping:
        for j in range(1, trunc_num):
            t1 = 4 * b * j * (j * c - a_bar) - (y - 2 * j * c) ** 2 / (2 * T)
            t2 = 4 * b * j * (j * c + a_bar) - (y + 2 * j * c) ** 2 / (2 * T)
            t3 = 2 * b * (2 * j - 1) * (j * c - a1) - (y + 2 * j * c - 2 * a1) ** 2 / (2 * T)
            t4 = 2 * b * (2 * j - 1) * (j * c + a2) - (y - 2 * j * c - 2 * a2) ** 2 / (2 * T)
            term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
            if fabs(term) < threshold:
                break
            result += term / sqrt(2 * M_PI)
    else:
        for j in range(1, trunc_num):
            t1 = 4 * b * j * (j * c - a_bar) - (y - 2 * j * c) ** 2 / (2 * T)
            t2 = 4 * b * j * (j * c + a_bar) - (y + 2 * j * c) ** 2 / (2 * T)
            t3 = 2 * b * (2 * j - 1) * (j * c - a1) - (y + 2 * j * c - 2 * a1) ** 2 / (2 * T)
            t4 = 2 * b * (2 * j - 1) * (j * c + a2) - (y - 2 * j * c - 2 * a2) ** 2 / (2 * T)
            term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
            result += term / sqrt(2 * M_PI)
    return result * factor

cpdef inline double fptd_single(double t, double mu, double sigma, double a1, double b1, double a2, double b2, double x0, int bdy, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True) noexcept nogil:
    mu = mu / sigma
    a1 = (a1 - x0) / sigma
    b1 = b1 / sigma
    a2 = (a2 - x0) / sigma
    b2 = b2 / sigma
    return fptd_basic(t, mu, a1, b1, a2, b2, bdy, trunc_num, threshold, adaptive_stopping)


cpdef inline double q_single(double x, double mu, double sigma, double a1, double b1, double a2, double b2, double T, double x0, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True) noexcept nogil:
    x = (x - x0) / sigma
    mu = mu / sigma
    a1 = (a1 - x0) / sigma
    b1 = b1 / sigma
    a2 = (a2 - x0) / sigma
    b2 = b2 / sigma
    return q_basic(x, mu, a1, b1, a2, b2, T, trunc_num, threshold, adaptive_stopping) / sigma


def log_fptd_basic(double t, double mu, double a1, double b1, double a2, double b2, int bdy, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True):
    """Safe log of :func:`fptd_basic`."""
    return positive_log(
        fptd_basic(t, mu, a1, b1, a2, b2, bdy, trunc_num, threshold, adaptive_stopping)
    )


def log_q_basic(double x, double mu, double a1, double b1, double a2, double b2, double T, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True):
    """Safe log of :func:`q_basic`."""
    return positive_log(
        q_basic(x, mu, a1, b1, a2, b2, T, trunc_num, threshold, adaptive_stopping)
    )


def log_fptd_single(double t, double mu, double sigma, double a1, double b1, double a2, double b2, double x0, int bdy, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True):
    """Safe log of :func:`fptd_single`."""
    return positive_log(
        fptd_single(
            t,
            mu,
            sigma,
            a1,
            b1,
            a2,
            b2,
            x0,
            bdy,
            trunc_num,
            threshold,
            adaptive_stopping,
        )
    )


def log_q_single(double x, double mu, double sigma, double a1, double b1, double a2, double b2, double T, double x0, int trunc_num=DEFAULT_TRUNC_NUM, double threshold=DEFAULT_THRESHOLD, bint adaptive_stopping=True):
    """Safe log of :func:`q_single`."""
    return positive_log(
        q_single(
            x,
            mu,
            sigma,
            a1,
            b1,
            a2,
            b2,
            T,
            x0,
            trunc_num,
            threshold,
            adaptive_stopping,
        )
    )
