"""Piecewise and parametric boundary functions for drift-diffusion models."""

import numpy as np


def piecewise_const_func(t, mu_array, node_array):
    """Piecewise constant drift rate function.

    Assumes pre-validated inputs (lengths match, node_array strictly
    increasing with node_array[0] == 0).  Validation is handled by
    :func:`~efpt.validation.check_multistage_params` at the
    model/API level.
    """
    d = len(mu_array)
    _node_array = np.append(node_array, np.inf)
    return np.piecewise(
        t,
        [(t >= _node_array[i]) & (t < _node_array[i + 1]) for i in range(d)],
        mu_array,
    )


def piecewise_linear_func(t, a_array, b_array, node_array):
    """Piecewise linear function.

    Assumes pre-validated inputs (lengths match, node_array strictly
    increasing with node_array[0] == 0).  Validation is handled by
    :func:`~efpt.validation.check_multistage_params` at the
    model/API level.
    """
    d = len(b_array)
    _node_array = np.append(node_array, np.inf)
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
