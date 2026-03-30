"""Generic numerical utilities for efficient-fpt."""

from __future__ import annotations

import warnings

import numpy as np

# Backward-compatible re-exports — callers that use
# ``from efficient_fpt.utils import ...`` will keep working.


def adaptive_interpolation(
    f,
    x_range,
    error_threshold,
    max_iterations=1000,
    initial_points=10,
    num_eval_points=1000,
):
    """
    Adaptive linear interpolation of a function `f` over a specified range `x_range`.
    The function iteratively refines the interpolation points until the maximum error
    on the evaluation grid is below the specified `error_threshold` or the maximum number of iterations is reached.
    The function returns the x-coordinates and corresponding y-coordinates of the interpolation.
    """
    x_points = np.linspace(x_range[0], x_range[1], initial_points)
    y_points = f(x_points)
    xi = np.linspace(x_range[0], x_range[1], num_eval_points)
    yi = np.interp(xi, x_points, y_points)
    iteration = 0
    while iteration < max_iterations:
        f_actual = f(xi)
        errors = np.abs(yi - f_actual)
        max_error = np.max(errors)
        if max_error <= error_threshold:
            break
        max_error_idx = np.argmax(errors)
        new_x = xi[max_error_idx]
        if np.any(np.isclose(x_points, new_x, atol=1e-12)):
            break
        new_y = f(new_x)
        idx = np.searchsorted(x_points, new_x)
        x_points = np.insert(x_points, idx, new_x)
        y_points = np.insert(y_points, idx, new_y)
        yi = np.interp(xi, x_points, y_points)
        iteration += 1
    else:
        warnings.warn(
            "Maximum iterations reached before meeting error threshold.",
            RuntimeWarning,
            stacklevel=2,
        )
    return x_points, y_points
