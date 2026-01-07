from numpy import exp, sqrt, pi
import numpy as np

"""
Reference:
Hall, W. J. (1997). The distribution of Brownian motion on linear stopping boundaries. Sequential analysis, 16(4), 345-352.
"""


def fptd_basic(t, mu, a1, b1, a2, b2, bdy, trunc_num=100, threshold=1e-20, fixed_terms=False):
    """
    First passage time density of Brownian motion with drift starting at x0 = 0 to
    the upper boundary u(t) = a1 + b1 * t and the lower boundary l(t) = a2 + b2 * t
    where a1 > 0 > a2, b1 < 0 < b2
    Note: set a1 = -a2 = a, b1 = b2 = -b to recover the symmetric case
    t shoud be in (0, -(a1 - a2) / (b1 - b2)), otherwise the density is 0
    
    Parameters
    ----------
    fixed_terms : bool, optional (default=False)
        If True, always compute exactly `trunc_num` terms without early termination.
        Useful for testing equivalence with JAX implementation.
    """
    a_bar = (a1 + a2) / 2
    b = (b2 - b1) / 2
    c = a1 - a2

    if bdy == 1:
        delta = mu - b1
        factor = t ** (-1.5) / sqrt(2 * pi) * exp(-b / c * a1**2 + a1 * delta - 0.5 * delta**2 * t)
    elif bdy == -1:
        delta = -mu + b2
        factor = t ** (-1.5) / sqrt(2 * pi) * exp(-b / c * a2**2 - a2 * delta - 0.5 * delta**2 * t)
    else:
        raise ValueError("bdy must be 1 or -1")
    result = 0
    for j in range(trunc_num):
        rj = (j + 0.5) * c + bdy * (-1) ** j * a_bar
        term = (-1) ** j * rj * exp((b / c - 1 / (2 * t)) * rj**2)
        if not fixed_terms and np.max(np.abs(term)) < threshold:
            break
        result += term
    return result * factor


def q_basic(x, mu, a1, b1, a2, b2, T, trunc_num=100, threshold=1e-20, fixed_terms=False):
    """
    density of Brownian motion with drift at time T starting at x0 = 0 
    given that it hasn't hit the upper boundary u(t) = a1 + b1 * t or the lower boundary l(t) = a2 + b2 * t
    upper boundary: u(t) = a1 + b1 * t
    lower boundary: l(t) = a2 + b2 * t
    vertical boundary: v(x) = T
    where a1 > 0 > a2, b1 < 0 < b2, T > 0
    Note: set a1 = -a2 = a, b1 = b2 = -b to recover the symmetric case
    x shoud be in (l(T), u(T)), otherwise the density is 0
    
    Parameters
    ----------
    fixed_terms : bool, optional (default=False)
        If True, always compute exactly `trunc_num` terms without early termination.
        Useful for testing equivalence with JAX implementation.
    """
    a_bar = (a1 + a2) / 2
    b = (b2 - b1) / 2
    b_bar = (b1 + b2) / 2
    c = a1 - a2
    y = x - b_bar * T
    factor = exp((mu - b_bar) * x - 0.5 * (mu**2 - b_bar**2) * T) / sqrt(T)
    result = 1 / sqrt(2 * pi) * exp(-(y**2) / (2 * T))
    for j in range(1, trunc_num):
        t1 = 4 * b * j * (j * c - a_bar) - (y - 2 * j * c) ** 2 / (2 * T)
        t2 = 4 * b * j * (j * c + a_bar) - (y + 2 * j * c) ** 2 / (2 * T)
        t3 = 2 * b * (2 * j - 1) * (j * c - a1) - (y + 2 * j * c - 2 * a1) ** 2 / (2 * T)
        t4 = 2 * b * (2 * j - 1) * (j * c + a2) - (y - 2 * j * c - 2 * a2) ** 2 / (2 * T)
        term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
        if not fixed_terms and np.max(np.abs(term)) < threshold:
            break
        result += term / sqrt(2 * pi)
    return result * factor


def fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy, trunc_num=100, threshold=1e-20, fixed_terms=False):
    """
    First passage time density with sigma scaling.
    
    Parameters
    ----------
    fixed_terms : bool, optional (default=False)
        If True, always compute exactly `trunc_num` terms without early termination.
    """
    mu = mu / sigma
    a1 = (a1 - x0) / sigma
    b1 = b1 / sigma
    a2 = (a2 - x0) / sigma
    b2 = b2 / sigma
    return fptd_basic(t, mu, a1, b1, a2, b2, bdy, trunc_num, threshold, fixed_terms)


def q_single(x, mu, sigma, a1, b1, a2, b2, T, x0, trunc_num=100, threshold=1e-20, fixed_terms=False):
    """
    Non-exit probability density with sigma scaling.
    
    Parameters
    ----------
    fixed_terms : bool, optional (default=False)
        If True, always compute exactly `trunc_num` terms without early termination.
    """
    x = (x - x0) / sigma
    mu = mu / sigma
    a1 = (a1 - x0) / sigma
    b1 = b1 / sigma
    a2 = (a2 - x0) / sigma
    b2 = b2 / sigma
    return q_basic(x, mu, a1, b1, a2, b2, T, trunc_num, threshold, fixed_terms) / sigma
