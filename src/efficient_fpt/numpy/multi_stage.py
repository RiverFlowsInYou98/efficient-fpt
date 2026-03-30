import numpy as np
from .single_stage import fptd_single, q_single
from ..validation import check_multistage_params
from .._defaults import DEFAULT_QUADRATURE_ORDER, DEFAULT_TRUNC_NUM, DEFAULT_THRESHOLD
from ..quadrature import lgwt_lookup_table


def _logsumexp(a, axis=None):
    """NumPy-only logsumexp specialized for the repo's reduction patterns."""
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        if a.size == 0:
            return -np.inf
        max_val = np.max(a)
        if np.isneginf(max_val):
            return -np.inf
        with np.errstate(invalid="ignore"):
            return max_val + np.log(np.sum(np.exp(a - max_val)))

    max_val = np.max(a, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        out = max_val + np.log(np.sum(np.exp(a - max_val), axis=axis, keepdims=True))
    out = np.where(np.isneginf(max_val), -np.inf, out)
    return np.squeeze(out, axis=axis)


def _positive_logs(a):
    """Return log(a) for positive entries and -inf otherwise, without warnings."""
    a = np.asarray(a, dtype=np.float64)
    logs = np.full_like(a, -np.inf, dtype=np.float64)
    positive = a > 0
    if np.any(positive):
        logs[positive] = np.log(a[positive])
    return logs


def compute_homog_multistage_fptds_and_npd(
    t_grid,
    T,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    eps=1e-3,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
    log_space=False,
):
    """
    Computes the first-passage time density (FPTD) for a multistage drift-diffusion model (DDM)
    on a specified time grid `t_grid`, for both upper and lower absorbing boundaries.

    Parameters
    ----------
    t_grid : array-like
        Time grid at which to evaluate the first-passage time density.

    mu_array : array-like of shape (d,)
        Drift rates for each of the `d` stages.

    node_array : array-like of shape (d,)
        Start times of each stage. Must satisfy ``node_array[0] == 0``.

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

    order : int, optional (default=30)
        Quadrature order used in `lgwt_lookup_table`.

    eps : float, optional (default=1e-3)
        Tolerance for ignoring grid points in `t_grid` that are too close to `node_array`,
        to avoid numerical instability.

    trunc_num : int, optional (default=100)
        Maximum number of terms in the truncated series expansion used for single-stage computations.

    threshold : float, optional (default=1e-20)
        Early-stopping tolerance for the single-stage series expansion.

    adaptive_stopping : bool, optional (default=True)
        If True, stop early when the current term is smaller than `threshold`.
        If False, always compute exactly `trunc_num` terms.

    log_space : bool, optional (default=False)
        If True, use log-space computation to prevent underflow in deep
        multi-stage models. Uses logsumexp for numerically stable accumulation.

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
    mu_array = mu_array[node_array < T]
    sigma_array = sigma_array[node_array < T]
    b1_array = b1_array[node_array < T]
    b2_array = b2_array[node_array < T]
    node_array = node_array[node_array < T]
    d = len(mu_array)  # Number of stages

    ##### ASSERTIONS #####
    check_multistage_params(
        mu_array, node_array, sigma_array, a1, b1_array, a2, b2_array
    )
    ##### END OF ASSERTIONS #####

    # Initialize
    ub, lb = a1, a2
    if isinstance(x0, np.ndarray) and x0.ndim == 2:
        ws = x0[0]
        xs = x0[1]
        qs = np.ones_like(ws)
    elif callable(x0):
        xs, ws = lgwt_lookup_table(order, lb, ub)
        qs = x0(xs)

    xs_prev, ws_prev, qs_prev, ub_prev, lb_prev = xs, ws, qs, ub, lb
    _node_array = np.concatenate([node_array, [T]])

    # skip t that are too close to `node_array` to avoid numerical instability issues
    t_grid, indices, _ = filter_and_group(_node_array, t_grid, epsilon=eps)
    upper_densities = np.zeros_like(t_grid)
    lower_densities = np.zeros_like(t_grid)

    if log_space:
        # Initialize log-space: log_ws_qs = log(ws * qs)
        log_ws_qs = _positive_logs(ws_prev * qs_prev)

    for n in range(d):
        ub += b1_array[n] * (_node_array[n + 1] - _node_array[n])
        lb += b2_array[n] * (_node_array[n + 1] - _node_array[n])

        xs, ws = lgwt_lookup_table(order, lb, ub)
        P = q_single(
            xs[:, np.newaxis],
            mu_array[n],
            sigma_array[n],
            ub_prev,
            b1_array[n],
            lb_prev,
            b2_array[n],
            _node_array[n + 1] - _node_array[n],
            xs_prev,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )

        if len(indices[n]) > 0:
            U = fptd_single(
                t_grid[indices[n]][:, np.newaxis] - _node_array[n],
                mu_array[n],
                sigma_array[n],
                ub_prev,
                b1_array[n],
                lb_prev,
                b2_array[n],
                xs_prev,
                1,
                trunc_num=trunc_num,
                threshold=threshold,
                adaptive_stopping=adaptive_stopping,
            )
            L = fptd_single(
                t_grid[indices[n]][:, np.newaxis] - _node_array[n],
                mu_array[n],
                sigma_array[n],
                ub_prev,
                b1_array[n],
                lb_prev,
                b2_array[n],
                xs_prev,
                -1,
                trunc_num=trunc_num,
                threshold=threshold,
                adaptive_stopping=adaptive_stopping,
            )
            if log_space:
                # log-space accumulation for FPTD
                log_U = _positive_logs(U)
                log_L = _positive_logs(L)
                upper_densities[indices[n]] = np.exp(
                    _logsumexp(log_U + log_ws_qs[np.newaxis, :], axis=1)
                )
                lower_densities[indices[n]] = np.exp(
                    _logsumexp(log_L + log_ws_qs[np.newaxis, :], axis=1)
                )
            else:
                upper_densities[indices[n]] = np.sum(ws_prev * qs_prev * U, axis=1)
                lower_densities[indices[n]] = np.sum(ws_prev * qs_prev * L, axis=1)

        if log_space:
            log_P = _positive_logs(P)
            # log_qs[i] = logsumexp_j(log_P[i,j] + log_ws_qs[j])
            log_qs = _logsumexp(log_P + log_ws_qs[np.newaxis, :], axis=1)
            log_ws = _positive_logs(ws)
            log_ws_qs = log_ws + log_qs
            qs = np.exp(log_qs)
        else:
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
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    d = len(a) - 1

    # Filter: exclude x values within epsilon of any boundary in a
    dists = np.abs(x[:, np.newaxis] - a[np.newaxis, :])
    too_close = np.any(dists < epsilon, axis=1)
    keep = ~too_close

    # Assign each x to an interval via searchsorted (bins are [a[i], a[i+1]))
    bin_idx = np.searchsorted(a, x, side="right") - 1  # interval index
    in_range = (bin_idx >= 0) & (bin_idx < d)
    keep = keep & in_range

    filtered_x = x[keep]
    filtered_bins = bin_idx[keep]

    classified_indices = [[] for _ in range(d)]
    classified_x = [[] for _ in range(d)]
    for idx, (val, b) in enumerate(zip(filtered_x, filtered_bins)):
        classified_indices[b].append(idx)
        classified_x[b].append(val)

    return filtered_x, classified_indices, classified_x
