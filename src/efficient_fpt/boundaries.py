"""Piecewise and parametric boundary functions for drift-diffusion models."""

import numpy as np


def piecewise_const_func(t, mu_array, node_array):
    """
    piecewise constant drift rate function, with drift rates `mu_array` and change points `node_array`
    """
    d = len(mu_array)
    if len(node_array) != d:
        raise ValueError(f"node_array length {len(node_array)} != mu_array length {d}")
    if not all(i < j for i, j in zip(node_array, node_array[1:])):
        raise ValueError("node_array must be strictly increasing")
    if d >= 2 and node_array[1] <= 0:
        raise ValueError("node_array[1] must be positive")
    _node_array = np.append(node_array, np.inf)
    return np.piecewise(
        t,
        [(t >= _node_array[i]) & (t < _node_array[i + 1]) for i in range(d)],
        mu_array,
    )


def piecewise_linear_func(t, a_array, b_array, node_array):
    """
    piecewise linear function, with intercepts `a_array`, slopes `b_array` and change points `node_array`
    """
    d = len(b_array)
    if len(a_array) != d:
        raise ValueError(f"a_array length {len(a_array)} != b_array length {d}")
    if len(node_array) != d:
        raise ValueError(f"node_array length {len(node_array)} != b_array length {d}")
    if not all(i < j for i, j in zip(node_array, node_array[1:])):
        raise ValueError("node_array must be strictly increasing")
    if d >= 2 and node_array[1] <= 0:
        raise ValueError("node_array[1] must be positive")

    # Extend node_array to include boundaries for the piecewise function
    _node_array = np.append(node_array, np.inf)

    # Define the piecewise function
    conds = [(t >= _node_array[i]) & (t < _node_array[i + 1]) for i in range(d)]
    funcs = [
        lambda t, i=i: a_array[i] + b_array[i] * (t - _node_array[i]) for i in range(d)
    ]
    return np.piecewise(t, conds, funcs)


def weibull_survival(t=1, lbda=1, k=1):
    """boundary based on weibull survival function.

    Arguments
    ---------
        t (int, optional): Defaults to 1.
        lbda (int, optional): Defaults to 1.
        k (int, optional): Defaults to 1.

    Returns
    -------
        np.array: Array of boundary values, same length as t
    """
    return np.exp(-np.power(np.divide(t, lbda), k))
