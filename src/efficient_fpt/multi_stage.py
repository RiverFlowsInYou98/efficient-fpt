import numpy as np
from .single_stage import fptd_single, q_single
from .utils import check_valid_multistage_params


# Pre-populate with commonly used orders (computed once at import time).
_GAUSS_LEGENDRE_CACHE = {
    n: np.polynomial.legendre.leggauss(n)
    for n in (1, 2, 3, 4, 5, 6, 10, 20, 30)
}


def lgwtLookupTable(order, a, b):
    """Gauss-Legendre quadrature nodes and weights on the interval [a, b].

    Parameters
    ----------
    order : int
        The order of the Gauss-Legendre quadrature.
    a, b : float
        Integration limits.

    Returns
    -------
    x : np.ndarray
        Nodes on [a, b].
    w : np.ndarray
        Weights on [a, b].
    """
    if order not in _GAUSS_LEGENDRE_CACHE:
        _GAUSS_LEGENDRE_CACHE[order] = np.polynomial.legendre.leggauss(order)
    x, w = _GAUSS_LEGENDRE_CACHE[order]
    # Map from [-1, 1] to [a, b]
    x = x * (b - a) / 2 + (b + a) / 2
    w = w * (b - a) / 2
    return x, w


def get_multistage_densities(t_grid, mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array, T, x0, order=30, eps=1e-3, trunc_num=100, threshold=1e-20):
    """
    Computes the first-passage time density (FPTD) for a multistage drift-diffusion model (DDM) 
    on a specified time grid `t_grid`, for both upper and lower absorbing boundaries.

    Parameters
    ----------
    t_grid : array-like
        Time grid at which to evaluate the first-passage time density.

    mu_array : array-like of shape (d,)
        Drift rates for each of the `d` stages.

    sacc_array : array-like of shape (d,)
        Start times of each stage. Must satisfy `sacc_array[0] == 0`.

    sigma_array : array-like of shape (d,)
        Diffusion coefficients for each stage.

    a1 : float
        Initial position of the upper boundary at time 0.

    b1_array : array-like of shape (d,)
        Slopes of the upper boundary in each stage (boundary evolves linearly).

    a2 : float
        Initial position of the lower boundary at time 0.

    b2_array : array-like of shape (d,)
        Slopes of the lower boundary in each stage (boundary evolves linearly).

    T : float
        Final time of the simulation. This defines the end of the last stage.

    x0 : callable or 2D np.ndarray
        Initial distribution of the diffusion process:
        - If callable, represents a sub-probability density function p(x_0) over the initial state.
        - If a 2D array of shape (2, N), X(0) is a mixture of N point masses.
          where the first row contains weights and the second row contains support points.

    eps : float, optional (default=1e-3)
        Tolerance for ignoring grid points in `t_grid` that are too close to `sacc_array`, to avoid numerical instability.

    trunc_num : int, optional (default=100)
        Number of terms to keep in the truncated series expansion used for single-stage computations.

    Returns
    -------
    fptd : np.ndarray of shape (3, len(t_grid))
        A matrix where:
        - Row 0: Filtered time grid
        - Row 1: FPTD values at the upper boundary
        - Row 2: FPTD values at the lower boundary

    final_state : np.ndarray of shape (2, N)
        The final (post-last-stage) distribution over the process state.
        - Row 0: Grid of support points for the process state
        - Row 1: Corresponding probabilities (subdensity mass) at those points
    """
    # Check parameters
    mu_array = mu_array[sacc_array < T]
    sigma_array = sigma_array[sacc_array < T]
    b1_array = b1_array[sacc_array < T]
    b2_array = b2_array[sacc_array < T]
    sacc_array = sacc_array[sacc_array < T]
    d = len(mu_array)  # Number of stages
    ##### ASSERTIONS #####
    check_valid_multistage_params(mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array)
    ##### END OF ASSERTIONS #####
    # Initialize
    ub, lb = a1, a2
    if isinstance(x0, np.ndarray) and x0.ndim == 2:
        ws = x0[0]
        xs = x0[1]
        qs = np.ones_like(ws)
    elif callable(x0):
        xs, ws = lgwtLookupTable(order, lb, ub)
        qs = x0(xs)
    xs_prev, ws_prev, qs_prev, ub_prev, lb_prev = xs, ws, qs, ub, lb
    _sacc_array = np.concatenate([sacc_array, [T]])
    # skipping t that are too close to `sacc_array` to avoid numerical instability issues
    t_grid, indices, _ = filter_and_group(_sacc_array, t_grid, epsilon=eps)
    upper_densities = np.zeros_like(t_grid)
    lower_densities = np.zeros_like(t_grid)
    for n in range(d):
        ub += b1_array[n] * (_sacc_array[n + 1] - _sacc_array[n])
        lb += b2_array[n] * (_sacc_array[n + 1] - _sacc_array[n])
        xs, ws = lgwtLookupTable(order, lb, ub)
        P = q_single(xs[:, np.newaxis], mu_array[n], sigma_array[n], ub_prev, b1_array[n], lb_prev, b2_array[n], _sacc_array[n + 1] - _sacc_array[n], xs_prev, trunc_num, threshold)
        if len(indices[n]) > 0:
            U = fptd_single(t_grid[indices[n]][:, np.newaxis] - _sacc_array[n], mu_array[n], sigma_array[n], ub_prev, b1_array[n], lb_prev, b2_array[n], xs_prev, 1, trunc_num, threshold)
            L = fptd_single(t_grid[indices[n]][:, np.newaxis] - _sacc_array[n], mu_array[n], sigma_array[n], ub_prev, b1_array[n], lb_prev, b2_array[n], xs_prev, -1, trunc_num, threshold)
            upper_densities[indices[n]] = np.sum(ws_prev * qs_prev * U, axis=1)
            lower_densities[indices[n]] = np.sum(ws_prev * qs_prev * L, axis=1)
        qs = np.sum(ws_prev * qs_prev * P, axis=1)
        xs_prev, ws_prev, qs_prev, ub_prev, lb_prev = xs, ws, qs, ub, lb
    return np.vstack([t_grid, upper_densities, lower_densities]), np.vstack([xs, qs])


def filter_and_group(a, x, epsilon=1e-3):
    """
    Filters and groups values of `x` into open intervals defined by consecutive 
    elements in `a`, excluding any values within `epsilon` of the interval boundaries.

    Parameters:
    ----------
    a : array-like of shape (d + 1,)
        Sorted array defining `d` open intervals of the form (a[i], a[i+1]).
    
    x : array-like
        Sorted array of values to be filtered and assigned to the intervals defined by `a`.
    
    epsilon : float, optional (default=1e-3)
        Tolerance for excluding values that are too close to the interval boundaries.

    Returns:
    -------
    filtered_x : np.ndarray
        Array of values from `x` that lie strictly inside one of the intervals 
        (a[i], a[i+1]), excluding points near the boundaries.

    classified_indices : list of lists
        Each sublist contains indices of `filtered_x` that belong to interval i.

    classified_x : list of lists
        Each sublist contains the actual values from `x` that fall into interval i.
    """
    # Initialize pointers and result structure
    i = 0
    j = 0
    d = len(a) - 1
    classified_x = [[] for _ in range(d)]
    classified_indices = [[] for _ in range(d)]
    index = 0  # This index tracks positions in the filtered x
    filtered_x = []
    # Process each element in x
    while j < len(x):
        # Filter out x[j] if it's too close to a[i] or a[i+1]
        while i < d and x[j] > a[i + 1]:
            i += 1  # Move to the next interval if x[j] is outside the current range
        if i < d and (abs(x[j] - a[i]) < epsilon or abs(x[j] - a[i + 1]) < epsilon):
            j += 1  # Skip x[j] if it's within epsilon of a[i] or a[i+1]
            continue
        # If x[j] is in the interval (a[i], a[i+1])
        if i < d and a[i] < x[j] < a[i + 1]:
            filtered_x.append(x[j])
            classified_x[i].append(x[j])
            classified_indices[i].append(index)
            index += 1
        j += 1  # Move to the next x[j]
    return np.array(filtered_x), classified_indices, classified_x
