import numpy as np


def check_valid_multistage_params(mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array):
    """
    Check the validity of the parameters for the multi-stage model.
    """
    d = len(mu_array)
    assert len(sacc_array) == d
    assert len(sigma_array) == d
    assert len(b1_array) == d
    assert len(b2_array) == d
    assert all(np.diff(sacc_array) > 0)
    assert sacc_array[0] == 0
    if d >= 2:
        assert sacc_array[1] > 0


def get_alternating_mu_array(mu1, mu2, d, flag):
    """
    Generate a list of alternating drift rates with length `d`
    len(sacc_array) = d - 1, len(mu_array) = d
    `flag`=0: mu1 -> mu2 -> mu1 -> ...
    `flag`=1: mu2 -> mu1 -> mu2 -> ...
    """
    mu_array = []
    current_mu = mu2 if flag else mu1
    for i in range(d):
        mu_array.append(current_mu)
        current_mu = mu2 if current_mu == mu1 else mu1
    return np.array(mu_array)


def adaptive_interpolation(f, x_range, error_threshold, max_iterations=1000, initial_points=10, num_eval_points=1000):
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
        print("Warning: Maximum iterations reached before meeting error threshold.")
    return x_points, y_points
