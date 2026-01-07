import numpy as np
from .single_stage import fptd_single, q_single
from .utils import check_valid_multistage_params


def lgwtLookupTable(order, a, b):
    """
    Look up Gauss-Legendre quadrature nodes and weights on the interval [a, b] for a given order.
    `x_ref` and `w_ref` are nodes and weights on the reference interval [-1, 1],
    which can be obtained from `x_ref, w_ref = np.polynomial.legendre.leggauss(orders)`
    Parameters:
        order (int): The order of the Gauss-Legendre quadrature.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.

    Returns:
        x (np.ndarray): Nodes.
        w (np.ndarray): Weights.
    """
    if order == 1:
        x = np.array([0.0])
        w = np.array([2.0])
    elif order == 2:
        x = np.array([-0.5773502691896257, 0.5773502691896257])
        w = np.array([1.0, 1.0])
    elif order == 3:
        x = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
        w = np.array([0.5555555555555557, 0.8888888888888888, 0.5555555555555557])
    elif order == 4:
        x = np.array([-0.8611363115940526, -0.33998104358485626, 0.33998104358485626, 0.8611363115940526])
        w = np.array([0.3478548451374537, 0.6521451548625462, 0.6521451548625462, 0.3478548451374537])
    elif order == 5:
        x = np.array([-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664])
        w = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])
    elif order == 6:
        x = np.array([-0.93246951, -0.66120939, -0.23861919, 0.23861919, 0.66120939, 0.93246951])
        w = np.array([0.17132449, 0.36076157, 0.46791393, 0.46791393, 0.36076157, 0.17132449])
    elif order == 10:
        x = np.array([-0.97390653, -0.86506337, -0.67940957, -0.43339539, -0.14887434, 0.14887434, 0.43339539, 0.67940957, 0.86506337, 0.97390653])
        w = np.array([0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422, 0.26926672, 0.21908636, 0.14945135, 0.06667134])
    elif order == 20:
        x = np.array([-0.9931285991850949, -0.9639719272779138, -0.9122344282513258, -0.8391169718222188, -0.7463319064601508, -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.2277858511416451, -0.07652652113349734, 0.07652652113349734, 0.2277858511416451, 0.37370608871541955, 0.5108670019508271, 0.636053680726515, 0.7463319064601508, 0.8391169718222188, 0.9122344282513258, 0.9639719272779138, 0.9931285991850949])
        w = np.array([0.017614007139153273, 0.04060142980038622, 0.06267204833410944, 0.08327674157670467, 0.10193011981724026, 0.11819453196151825, 0.13168863844917653, 0.14209610931838187, 0.14917298647260366, 0.15275338713072578, 0.15275338713072578, 0.14917298647260366, 0.14209610931838187, 0.13168863844917653, 0.11819453196151825, 0.10193011981724026, 0.08327674157670467, 0.06267204833410944, 0.04060142980038622, 0.017614007139153273])
    elif order == 30:
        x = np.array([-0.9968934840746495, -0.9836681232797473, -0.9600218649683075, -0.9262000474292743, -0.8825605357920526, -0.8295657623827684, -0.7677774321048262, -0.6978504947933158, -0.6205261829892429, -0.5366241481420199, -0.44703376953808915, -0.3527047255308781, -0.25463692616788985, -0.15386991360858354, -0.0514718425553177, 0.0514718425553177, 0.15386991360858354, 0.25463692616788985, 0.3527047255308781, 0.44703376953808915, 0.5366241481420199, 0.6205261829892429, 0.6978504947933158, 0.7677774321048262, 0.8295657623827684, 0.8825605357920526, 0.9262000474292743, 0.9600218649683075, 0.9836681232797473, 0.9968934840746495])
        w = np.array([0.007968192496169523, 0.018466468311091087, 0.028784707883322873, 0.03879919256962679, 0.048402672830594434, 0.05749315621761909, 0.06597422988218032, 0.0737559747377048, 0.08075589522941981, 0.0868997872010827, 0.09212252223778579, 0.09636873717464399, 0.09959342058679493, 0.10176238974840521, 0.10285265289355848, 0.10285265289355848, 0.10176238974840521, 0.09959342058679493, 0.09636873717464399, 0.09212252223778579, 0.0868997872010827, 0.08075589522941981, 0.0737559747377048, 0.06597422988218032, 0.05749315621761909, 0.048402672830594434, 0.03879919256962679, 0.028784707883322873, 0.018466468311091087, 0.007968192496169523])
    else:
        raise ValueError("Order not supported")

    # Adjust nodes and weights to the interval [a, b]
    x = x * (b - a) / 2 + (b + a) / 2
    w = w * (b - a) / 2

    return x, w


def get_multistage_densities(t_grid, mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array, T, x0, eps=1e-3, trunc_num=100, threshold=1e-20, fixed_terms=False):
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

    fixed_terms : bool, optional (default=False)
        If True, always compute exactly `trunc_num` terms without early termination.
        Useful for testing equivalence with JAX implementation.

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
        xs, ws = lgwtLookupTable(30, lb, ub)
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
        if n < d - 2:
            xs, ws = lgwtLookupTable(30, lb, ub)
        elif n == d - 2:
            xs, ws = lgwtLookupTable(30, lb, ub)
        else:  # n == d - 1
            xs, ws = lgwtLookupTable(30, lb, ub)
        P = q_single(xs[:, np.newaxis], mu_array[n], sigma_array[n], ub_prev, b1_array[n], lb_prev, b2_array[n], _sacc_array[n + 1] - _sacc_array[n], xs_prev, trunc_num, threshold, fixed_terms)
        if len(indices[n]) > 0:
            U = fptd_single(t_grid[indices[n]][:, np.newaxis] - _sacc_array[n], mu_array[n], sigma_array[n], ub_prev, b1_array[n], lb_prev, b2_array[n], xs_prev, 1, trunc_num, threshold, fixed_terms)
            L = fptd_single(t_grid[indices[n]][:, np.newaxis] - _sacc_array[n], mu_array[n], sigma_array[n], ub_prev, b1_array[n], lb_prev, b2_array[n], xs_prev, -1, trunc_num, threshold, fixed_terms)
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
