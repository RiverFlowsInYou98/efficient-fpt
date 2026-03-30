"""JAX single-trial multi-stage first-passage time density computation.

This module holds *both* explicit single-trial JAX implementations:

- public production kernels:
  ``compute_addm_fptd_precomputed`` and
  ``compute_heterog_multistage_fptd_precomputed``
- public stage-scan kernels:
  ``compute_addm_fptd_stagescan`` and
  ``compute_heterog_multistage_fptd_stagescan``

The production path precomputes all stage transition matrices for one trial
with ``vmap`` over stage index, then propagates the alive-state mass through
those transitions. The stage-scan path computes transition matrices inside
``lax.scan`` as the stage recurrence advances. That path is slower, but it is
useful for correctness checks, benchmarking, and algorithm comparison.

Relationship to the other JAX modules
-------------------------------------
- ``jax.batch`` contains the batched aDDM kernels. Its legacy baseline vmaps
  the public single-trial production function in this module over trials,
  while its production batch kernel carries the whole batch through one
  dedicated stage scan and computes transitions on the fly stage by stage.

So the split is:
- this file: single-trial precomputed + single-trial stage-scan
- ``jax.batch``: batch legacy baseline + batch production

The plain aliases ``compute_addm_fptd`` and
``compute_heterog_multistage_fptd`` continue to point to the precomputed
production kernels.

JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
``adaptive_stopping`` or ``threshold`` option in this backend.
"""

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import logsumexp

from .addm_helpers import _build_addm_mu_array
from .single_stage import fptd_single, q_single
from .utils import get_gauss_legendre_ref, _DUMMY_STAGE_DURATION
from .._defaults import DEFAULT_QUADRATURE_ORDER, DEFAULT_TRUNC_NUM


# ---------------------------------------------------------------------------
# Tiny numeric helpers
# ---------------------------------------------------------------------------


def _to_log_space(a):
    """Return log(a) for positive entries and -inf otherwise."""
    safe = jnp.where(a > 0, a, 1.0)
    return jnp.where(a > 0, jnp.log(safe), -jnp.inf)


def _safe_stage_durations(node_array, d):
    """Build a numerically safe duration vector from a padded stage-time array.

    Parameters
    ----------
    node_array : jax.Array, shape (max_d,)
        Padded stage start times. Only the first ``d`` entries are treated as
        valid stage onsets; later entries may be arbitrary padding.
    d : int
        Number of valid stages.

    Returns
    -------
    valid_stage_mask : jax.Array, shape (max_d - 1,)
        Boolean mask marking which entries of ``diff(node_array)`` correspond to
        real stage-to-stage durations. For ``d`` valid stages there are exactly
        ``d - 1`` valid transitions.
    safe_stage_duration_array : jax.Array, shape (max_d - 1,)
        Per-transition duration array used by the JAX multistage kernels.
        Valid durations are passed through unchanged. Inactive padded
        transitions are replaced by ``_DUMMY_STAGE_DURATION`` so traced JAX code
        never sees padded zero or negative stage lengths.

    Notes
    -----
    This helper exists because the JAX kernels operate on fixed-size padded
    arrays. Even inactive padded stages still participate in tracing, so they
    must be given harmless positive durations.

    Example
    -------
    If ``node_array = [0, 1, 3, 7, 0, 0, 0]`` and ``d = 4``, then the valid
    stages start at times ``0, 1, 3, 7``. The real stage durations are
    ``[1, 2, 4]`` and the padded tail is ignored:

    ``valid_stage_mask = [True, True, True, False, False, False]``

    ``safe_stage_duration_array = [1, 2, 4, dummy, dummy, dummy]``
    """
    max_d = node_array.shape[0]
    dtype = node_array.dtype
    if max_d <= 1:
        empty = jnp.empty((0,), dtype=dtype)
        return empty, empty

    raw_stage_durations = jnp.diff(node_array)
    stage_idx = jnp.arange(max_d - 1)
    valid_stage_mask = stage_idx < (d - 1)
    safe_stage_duration_array = jnp.where(
        valid_stage_mask,
        raw_stage_durations,
        _DUMMY_STAGE_DURATION,
    )
    return valid_stage_mask, safe_stage_duration_array


# ---------------------------------------------------------------------------
# Schedule builders
# ---------------------------------------------------------------------------


def _effective_addm_schedule(sacc_array, d, a, b):
    """Construct the stage schedule for the symmetric aDDM boundary geometry.

    Parameters
    ----------
    sacc_array : jax.Array, shape (max_d,)
        Padded aDDM stage onset times.
    d : int
        Number of valid stages.
    a : float
        Initial upper-boundary intercept. The lower boundary starts at ``-a``.
    b : float
        Symmetric boundary-collapse slope magnitude. The upper boundary slope is
        ``-b`` and the lower boundary slope is ``+b`` on active stages.

    Returns
    -------
    safe_stage_duration_array : jax.Array, shape (max_d - 1,)
        Safe per-stage durations from :func:`_safe_stage_durations`.
    upper_slope_array : jax.Array, shape (max_d - 1,)
        Upper-boundary slope per transition. Active entries are ``-b`` and
        inactive padded entries are ``0``.
    lower_slope_array : jax.Array, shape (max_d - 1,)
        Lower-boundary slope per transition. Active entries are ``+b`` and
        inactive padded entries are ``0``.
    a_starts : jax.Array, shape (max_d,)
        Upper-boundary value at the start of each stage. Because the aDDM
        boundary is symmetric, the lower-boundary start is always ``-a_starts``.

    Notes
    -----
    The multistage aDDM kernel uses ``a_starts[k]`` to place quadrature nodes at
    the start of stage ``k`` and uses ``upper_slope_array[k]`` / ``lower_slope_array[k]``
    to propagate the boundaries through the corresponding duration.

    Example
    -------
    If ``sacc_array = [0, 1, 3, 7, 0, 0, 0]``, ``d = 4``, ``a = 1.5``, and
    ``b = 0.3``, then:

    ``safe_stage_duration_array = [1, 2, 4, dummy, dummy, dummy]``

    ``upper_slope_array = [-0.3, -0.3, -0.3, 0, 0, 0]``

    ``lower_slope_array = [0.3, 0.3, 0.3, 0, 0, 0]``

    ``a_starts = [1.5, 1.2, 0.6, -0.6, -0.6, -0.6, -0.6]``

    The repeated tail means the padded stages are inert: once the valid stages
    end, the boundary start stays frozen.
    """
    dtype = sacc_array.dtype
    valid_stage_mask, safe_stage_duration_array = _safe_stage_durations(sacc_array, d)
    upper_slope_array = jnp.where(valid_stage_mask, -b, 0.0)
    lower_slope_array = jnp.where(valid_stage_mask, b, 0.0)
    a_starts = jnp.concatenate(
        [
            jnp.full((1,), a, dtype=dtype),
            a + jnp.cumsum(upper_slope_array * safe_stage_duration_array),
        ]
    )
    return safe_stage_duration_array, upper_slope_array, lower_slope_array, a_starts


def _effective_general_schedule(node_array, d, a1, b1_array, a2, b2_array):
    """Construct the stage schedule for the general asymmetric multistage model.

    Parameters
    ----------
    node_array : jax.Array, shape (max_d,)
        Padded stage onset times.
    d : int
        Number of valid stages.
    a1, a2 : float
        Upper and lower boundary intercepts at the start of stage 0.
    b1_array, b2_array : jax.Array, shape (max_d,)
        Per-stage upper and lower boundary slopes. Only the first ``d`` entries
        are meaningful.

    Returns
    -------
    safe_stage_duration_array : jax.Array, shape (max_d - 1,)
        Safe per-stage durations from :func:`_safe_stage_durations`.
    upper_slope_array : jax.Array, shape (max_d - 1,)
        Active upper-boundary slopes copied from ``b1_array[:-1]``; padded
        entries are set to ``0``.
    lower_slope_array : jax.Array, shape (max_d - 1,)
        Active lower-boundary slopes copied from ``b2_array[:-1]``; padded
        entries are set to ``0``.
    ub_starts : jax.Array, shape (max_d,)
        Upper boundary value at the start of each stage.
    lb_starts : jax.Array, shape (max_d,)
        Lower boundary value at the start of each stage.

    Notes
    -----
    This is the generalized analogue of :func:`_effective_addm_schedule`.
    Unlike the aDDM helper, upper and lower boundaries are tracked separately.

    Example
    -------
    If ``node_array = [0, 1, 3, 7, 0, 0, 0]``, ``d = 4``,
    ``a1 = 1.5``, ``a2 = -1.5``,
    ``b1_array = [-0.3, -0.1, 0.0, 0, 0, 0, 0]``, and
    ``b2_array = [0.2, 0.4, 0.1, 0, 0, 0, 0]``, then:

    ``safe_stage_duration_array = [1, 2, 4, dummy, dummy, dummy]``

    ``upper_slope_array = [-0.3, -0.1, 0.0, 0, 0, 0]``

    ``lower_slope_array = [0.2, 0.4, 0.1, 0, 0, 0]``

    ``ub_starts = [1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0]``

    ``lb_starts = [-1.5, -1.3, -0.5, -0.1, -0.1, -0.1, -0.1]``
    """
    dtype = node_array.dtype
    valid_stage_mask, safe_stage_duration_array = _safe_stage_durations(node_array, d)
    upper_slope_array = jnp.where(valid_stage_mask, b1_array[:-1], 0.0)
    lower_slope_array = jnp.where(valid_stage_mask, b2_array[:-1], 0.0)
    ub_starts = jnp.concatenate(
        [
            jnp.full((1,), a1, dtype=dtype),
            a1 + jnp.cumsum(upper_slope_array * safe_stage_duration_array),
        ]
    )
    lb_starts = jnp.concatenate(
        [
            jnp.full((1,), a2, dtype=dtype),
            a2 + jnp.cumsum(lower_slope_array * safe_stage_duration_array),
        ]
    )
    return (
        safe_stage_duration_array,
        upper_slope_array,
        lower_slope_array,
        ub_starts,
        lb_starts,
    )


# ---------------------------------------------------------------------------
# Shared precomputed-transition propagation helper
# ---------------------------------------------------------------------------


def _propagate_ws_pv(P_all, ws_all, ws_pv_init, d, log_space):
    """Propagate ws_pv across precomputed stage transitions.
    ws_pv = ws * pv (elementwise product),
    where ws is the quadrature weight and pv is non-passive density.

    Parameters
    ----------
    P_all : jax.Array, shape (max_d - 2, order, order)
        Precomputed transition matrices between consecutive stage interfaces.
    ws_all : jax.Array, shape (order, max_d - 1)
        Quadrature weights at each stage interface.
    ws_pv_init : jax.Array, shape (order,)
        Weighted alive-state mass after the first stage.
    d : int
        Number of valid stages.
    log_space : bool
        Whether ``ws_pv_init`` and the propagated state are represented in log
        space.

    Returns
    -------
    jax.Array, shape (order,)
        Weighted alive-state mass at the start of the final valid stage.

    Notes
    -----
    The scan still iterates over the padded tail for fixed-shape JAX tracing,
    but once ``k >= d - 2`` the carry is left unchanged.
    """
    stage_indices = jnp.arange(P_all.shape[0])

    if log_space:
        log_P_all = _to_log_space(P_all)

        def mv_step(log_ws_pv_prev, k):
            log_pv_new = logsumexp(log_P_all[k] + log_ws_pv_prev[None, :], axis=1)
            log_ws_pv_new = _to_log_space(ws_all[:, k + 1]) + log_pv_new
            active = k < (d - 2)
            return jnp.where(active, log_ws_pv_new, log_ws_pv_prev), None

    else:

        def mv_step(ws_pv_prev, k):
            ws_k = ws_all[:, k + 1]
            pv_new = P_all[k] @ ws_pv_prev
            ws_pv_new = ws_k * pv_new
            active = k < (d - 2)
            return jnp.where(active, ws_pv_new, ws_pv_prev), None

    ws_pv_final, _ = lax.scan(mv_step, ws_pv_init, stage_indices)
    return ws_pv_final


# ---------------------------------------------------------------------------
# Single-trial production kernels: precomputed transitions
# ---------------------------------------------------------------------------


def _addm_fptd_precomputed(
    rt,
    choice,
    sigma,
    a,
    b,
    x0,
    mu_array,
    sacc_array,
    d,
    *,
    order,
    trunc_num,
    log_space,
):
    """Single-trial ADDM FPTD with precomputed transition matrices.

    The computation skeleton is:

    1. build the per-stage boundary schedule
    2. place quadrature nodes and weights at stage interfaces
    3. propagate alive-state mass through intermediate stages
    4. evaluate final-stage boundary-hit densities
    5. reduce over latent start positions in the final stage

    Notes
    -----
    This is the production single-trial JAX kernel. For one trial, it
    precomputes the intermediate stage transition matrices with ``vmap`` over
    stage index, then propagates the alive-state quadrature mass through them.

    This should be contrasted with:

    - ``compute_addm_fptd_stagescan`` below in this same module:
      single-trial, but computes transitions inside ``lax.scan``
    - ``jax.batch.compute_addm_likelihoods_batchvmap``: batch baseline that vmaps
      the public single-trial API over trials
    - ``jax.batch.compute_addm_likelihoods_batchscan``: production batch kernel
      that carries the whole batch through one scan and computes transitions
      on the fly stage by stage
    """
    # Reference Gauss-Legendre quadrature on [-1, 1].
    x_ref, w_ref = get_gauss_legendre_ref(order)

    max_d = mu_array.shape[0]
    if max_d < 2:
        return fptd_single(
            rt, mu_array[0], sigma, a, -b, -a, b, x0, choice, trunc_num=trunc_num
        )

    # Build stage-local boundary geometry, then place interface quadrature
    # nodes/weights for every potential stage transition.
    safe_stage_duration_array, upper_slope_array, lower_slope_array, a_starts = (
        _effective_addm_schedule(sacc_array, d, a, b)
    )
    # xs_all / ws_all have shape (order, max_d - 1). Column k is the latent
    # state quadrature grid at the start of stage k + 1.
    xs_all = x_ref[:, None] * a_starts[1:]
    ws_all = w_ref[:, None] * a_starts[1:]

    # Propagate the initial point x0 through stage 0 to the first interface.
    pv_init = q_single(
        xs_all[:, 0],
        mu_array[0],
        sigma,
        a,
        upper_slope_array[0],
        -a,
        lower_slope_array[0],
        safe_stage_duration_array[0],
        x0,
        trunc_num=trunc_num,
    )

    if log_space:
        ws_pv_init = _to_log_space(ws_all[:, 0] * pv_init)
    else:
        ws_pv_init = ws_all[:, 0] * pv_init

    if max_d > 2:

        def compute_transition_matrices(k):
            stage_idx = k + 1
            a_prev = a_starts[stage_idx]
            return q_single(
                xs_all[:, stage_idx, None],
                mu_array[stage_idx],
                sigma,
                a_prev,
                upper_slope_array[stage_idx],
                -a_prev,
                lower_slope_array[stage_idx],
                safe_stage_duration_array[stage_idx],
                xs_all[:, k, None].T,
                trunc_num=trunc_num,
            )

        # P_all has shape (max_d - 2, order, order). Entry k maps interface k
        # to interface k + 1 for this trial.
        P_all = vmap(compute_transition_matrices)(jnp.arange(max_d - 2))
        ws_pv_final = _propagate_ws_pv(P_all, ws_all, ws_pv_init, d, log_space)
    else:
        ws_pv_final = ws_pv_init

    # Select the real final stage, evaluate boundary-hit density from every
    # latent start position, then quadrature-reduce to a scalar likelihood.
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)
    quad_idx = jnp.maximum(safe_d_idx - 1, 0)
    xs_final = xs_all[:, quad_idx]
    a_final = a_starts[safe_d_idx]
    sacc_final = sacc_array[safe_d_idx]
    mu_final = mu_array[safe_d_idx]
    t_in_final_stage = rt - sacc_final

    fptds = fptd_single(
        t_in_final_stage,
        mu_final,
        sigma,
        a_final,
        -b,
        -a_final,
        b,
        xs_final,
        choice,
        trunc_num=trunc_num,
    )

    if log_space:
        return jnp.exp(logsumexp(_to_log_space(fptds) + ws_pv_final))
    return jnp.sum(fptds * ws_pv_final)


def compute_addm_fptd_precomputed(
    rt,
    choice,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1,
    r2,
    flag,
    sacc_array,
    d,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Multi-stage FPTD for attention-dependent drift diffusion model.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
    ``adaptive_stopping`` or ``threshold`` option in this backend.

    This public wrapper keeps the high-level addm signature
    ``(eta, kappa, r1, r2, flag)`` and dispatches to the explicit production
    kernel after building the derived stage drifts.
    """
    mu_array = _build_addm_mu_array(
        eta,
        kappa,
        r1,
        r2,
        flag,
        d,
        sacc_array.shape[0],
        sacc_array.dtype,
    )

    def single_fn(_):
        return fptd_single(
            rt,
            mu_array[0],
            sigma,
            a,
            -b,
            -a,
            b,
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _addm_fptd_precomputed(
            rt,
            choice,
            sigma,
            a,
            b,
            x0,
            mu_array,
            sacc_array,
            d,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


# ---------------------------------------------------------------------------
# Generalized single-trial production kernel: precomputed transitions
# ---------------------------------------------------------------------------


def _heterog_multistage_fptd_precomputed(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order,
    trunc_num,
    log_space,
):
    """Generalized multi-stage FPTD with precomputed transitions.

    The computation skeleton mirrors :func:`_addm_fptd_precomputed`, but with
    separate upper/lower boundary schedules and per-stage sigma values.
    """
    # Reference Gauss-Legendre quadrature on [-1, 1].
    x_ref, w_ref = get_gauss_legendre_ref(order)

    max_d = mu_array.shape[0]
    if max_d < 2:
        return fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )

    (
        safe_stage_duration_array,
        upper_slope_array,
        lower_slope_array,
        ub_starts,
        lb_starts,
    ) = _effective_general_schedule(node_array, d, a1, b1_array, a2, b2_array)

    # xs_all / ws_all have shape (order, max_d - 1). Each column is the
    # quadrature grid at the start of the next stage.
    half_w_starts = (ub_starts[1:] - lb_starts[1:]) / 2.0
    center_starts = (ub_starts[1:] + lb_starts[1:]) / 2.0
    xs_all = x_ref[:, None] * half_w_starts + center_starts
    ws_all = w_ref[:, None] * half_w_starts

    # Propagate x0 through stage 0 to the first interface.
    pv_init = q_single(
        xs_all[:, 0],
        mu_array[0],
        sigma_array[0],
        a1,
        upper_slope_array[0],
        a2,
        lower_slope_array[0],
        safe_stage_duration_array[0],
        x0,
        trunc_num=trunc_num,
    )

    if log_space:
        ws_pv_init = _to_log_space(ws_all[:, 0] * pv_init)
    else:
        ws_pv_init = ws_all[:, 0] * pv_init

    if max_d > 2:

        def compute_transition_matrices(k):
            stage_idx = k + 1
            return q_single(
                xs_all[:, k + 1, None],
                mu_array[stage_idx],
                sigma_array[stage_idx],
                ub_starts[stage_idx],
                upper_slope_array[stage_idx],
                lb_starts[stage_idx],
                lower_slope_array[stage_idx],
                safe_stage_duration_array[stage_idx],
                xs_all[:, k, None].T,
                trunc_num=trunc_num,
            )

        # P_all has shape (max_d - 2, order, order).
        P_all = vmap(compute_transition_matrices)(jnp.arange(max_d - 2))
        ws_pv_final = _propagate_ws_pv(P_all, ws_all, ws_pv_init, d, log_space)
    else:
        ws_pv_final = ws_pv_init

    # Select the true final stage, evaluate the final-stage hit density from
    # each latent start position, then reduce over quadrature nodes.
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)
    quad_idx = jnp.maximum(safe_d_idx - 1, 0)
    xs_final = xs_all[:, quad_idx]
    ub_final = ub_starts[safe_d_idx]
    lb_final = lb_starts[safe_d_idx]
    node_final = node_array[safe_d_idx]
    mu_final = mu_array[safe_d_idx]
    sigma_final = sigma_array[safe_d_idx]
    b1_final = b1_array[safe_d_idx]
    b2_final = b2_array[safe_d_idx]
    t_in_final_stage = rt - node_final

    fptds = fptd_single(
        t_in_final_stage,
        mu_final,
        sigma_final,
        ub_final,
        b1_final,
        lb_final,
        b2_final,
        xs_final,
        choice,
        trunc_num=trunc_num,
    )

    if log_space:
        return jnp.exp(logsumexp(_to_log_space(fptds) + ws_pv_final))
    return jnp.sum(fptds * ws_pv_final)


def compute_heterog_multistage_fptd_precomputed(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Generalized multi-stage FPTD with per-stage sigma and boundary slopes.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
    ``adaptive_stopping`` or ``threshold`` option in this backend.

    This public wrapper keeps the generalized multistage signature while
    dispatching to the explicit precomputed production kernel.
    """

    def single_fn(_):
        return fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _heterog_multistage_fptd_precomputed(
            rt,
            choice,
            x0,
            a1,
            a2,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            d,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


# ---------------------------------------------------------------------------
# Single-trial stage-scan kernels
# ---------------------------------------------------------------------------


def _addm_fptd_stagescan(
    rt,
    choice,
    sigma,
    a,
    b,
    x0,
    mu_array,
    sacc_array,
    d,
    *,
    order,
    trunc_num,
    log_space,
):
    """Single-trial ADDM stage-scan kernel using scan-time transition updates.

    The computation skeleton is:

    1. build the per-stage boundary schedule
    2. place quadrature nodes and weights at the first interface
    3. advance stage by stage inside ``lax.scan``
    4. evaluate the final-stage boundary-hit density
    5. reduce over latent start positions in the final stage

    Notes
    -----
    This is the public single-trial stage-scan implementation for JAX
    multistage ADDM. Unlike :func:`_addm_fptd_precomputed`, it does not
    materialize all stage transition matrices up front. Instead, each stage
    transition matrix is computed inside the ``lax.scan`` body and consumed
    immediately.

    That makes this path a useful correctness oracle:

    - same backend as the production JAX path
    - simpler stage-by-stage recurrence
    - lower conceptual coupling to precomputed ``P_all``

    It is usually slower than the production precomputed kernel and is kept as
    the explicit stage-scan alternative for tests, benchmarking, and clearer
    algorithmic comparison.
    """
    # Reference Gauss-Legendre quadrature on [-1, 1].
    x_ref, w_ref = get_gauss_legendre_ref(order)

    max_d = mu_array.shape[0]
    if max_d < 2:
        return fptd_single(
            rt, mu_array[0], sigma, a, -b, -a, b, x0, choice, trunc_num=trunc_num
        )

    # Build stage-local geometry, then place the first interface quadrature
    # nodes/weights explicitly.
    safe_stage_duration_array, upper_slope_array, lower_slope_array, a_starts = (
        _effective_addm_schedule(sacc_array, d, a, b)
    )

    a_1 = a_starts[1]
    xs_init = x_ref * a_1
    ws_init = w_ref * a_1

    pv_init = q_single(
        xs_init,
        mu_array[0],
        sigma,
        a,
        upper_slope_array[0],
        -a,
        lower_slope_array[0],
        safe_stage_duration_array[0],
        x0,
        trunc_num=trunc_num,
    )

    if log_space:
        ws_pv_init = _to_log_space(ws_init * pv_init)
    else:
        ws_pv_init = ws_init * pv_init

    # Carry tracks the current interface nodes, weighted alive-state mass, and
    # boundary height at the start of the current stage.
    carry = (xs_init, ws_pv_init, a_1)

    if log_space:

        def stage_step(carry, stage_idx):
            xs_prev, log_ws_pv_prev, a_prev = carry
            a_curr = a_starts[stage_idx + 1]
            xs = x_ref * a_curr
            ws = w_ref * a_curr
            P = q_single(
                xs[:, None],
                mu_array[stage_idx],
                sigma,
                a_prev,
                upper_slope_array[stage_idx],
                -a_prev,
                lower_slope_array[stage_idx],
                safe_stage_duration_array[stage_idx],
                xs_prev[None, :],
                trunc_num=trunc_num,
            )
            log_pv = logsumexp(_to_log_space(P) + log_ws_pv_prev[None, :], axis=1)
            log_ws_pv = _to_log_space(ws) + log_pv
            active = stage_idx < (d - 1)
            xs_out = jnp.where(active, xs, xs_prev)
            ws_pv_out = jnp.where(active, log_ws_pv, log_ws_pv_prev)
            a_out = jnp.where(active, a_curr, a_prev)
            return (xs_out, ws_pv_out, a_out), None

    else:

        def stage_step(carry, stage_idx):
            xs_prev, ws_pv_prev, a_prev = carry
            a_curr = a_starts[stage_idx + 1]
            xs = x_ref * a_curr
            ws = w_ref * a_curr
            P = q_single(
                xs[:, None],
                mu_array[stage_idx],
                sigma,
                a_prev,
                upper_slope_array[stage_idx],
                -a_prev,
                lower_slope_array[stage_idx],
                safe_stage_duration_array[stage_idx],
                xs_prev[None, :],
                trunc_num=trunc_num,
            )
            pv = P @ ws_pv_prev
            ws_pv = ws * pv
            active = stage_idx < (d - 1)
            xs_out = jnp.where(active, xs, xs_prev)
            ws_pv_out = jnp.where(active, ws_pv, ws_pv_prev)
            a_out = jnp.where(active, a_curr, a_prev)
            return (xs_out, ws_pv_out, a_out), None

    if max_d > 2:
        carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 1))

    # Evaluate the observed hit density in the true final stage and reduce over
    # the latent start-position quadrature grid.
    xs_final, ws_pv_final, _ = carry
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)
    a_final = a_starts[safe_d_idx]
    sacc_final = sacc_array[safe_d_idx]
    mu_final = mu_array[safe_d_idx]
    t_in_final_stage = rt - sacc_final

    fptds = fptd_single(
        t_in_final_stage,
        mu_final,
        sigma,
        a_final,
        -b,
        -a_final,
        b,
        xs_final,
        choice,
        trunc_num=trunc_num,
    )

    if log_space:
        return jnp.exp(logsumexp(_to_log_space(fptds) + ws_pv_final))
    return jnp.sum(fptds * ws_pv_final)


def compute_addm_fptd_stagescan(
    rt,
    choice,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1,
    r2,
    flag,
    sacc_array,
    d,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Public single-trial ADDM stage-scan wrapper.

    This is the public stage-scan counterpart to
    :func:`compute_addm_fptd_precomputed`. Like the rest of the public aDDM
    surface, it accepts the original aDDM parameters and covariates
    ``(eta, kappa, r1, r2, flag)`` rather than a derived ``mu_array``.

    The plain :func:`compute_addm_fptd` alias continues to point to the
    precomputed production path.
    """
    mu_array = _build_addm_mu_array(
        eta,
        kappa,
        r1,
        r2,
        flag,
        d,
        sacc_array.shape[0],
        sacc_array.dtype,
    )

    def single_fn(_):
        return fptd_single(
            rt,
            mu_array[0],
            sigma,
            a,
            -b,
            -a,
            b,
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _addm_fptd_stagescan(
            rt,
            choice,
            sigma,
            a,
            b,
            x0,
            mu_array,
            sacc_array,
            d,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


def _heterog_multistage_fptd_stagescan(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order,
    trunc_num,
    log_space,
):
    """Single-trial generalized multistage stage-scan kernel.

    The computation skeleton mirrors :func:`_addm_fptd_stagescan`, but tracks
    separate upper/lower boundary positions and per-stage sigma values.

    Notes
    -----
    This is the public stage-scan counterpart to
    :func:`_heterog_multistage_fptd_precomputed`. It computes each stage
    transition matrix inside the scan body, which makes the execution order
    closer to the mathematical recurrence and easier to compare against other
    backends.
    """
    # Reference Gauss-Legendre quadrature on [-1, 1].
    x_ref, w_ref = get_gauss_legendre_ref(order)

    max_d = mu_array.shape[0]
    if max_d < 2:
        return fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )

    # Build stage-local geometry, then place the first interface quadrature
    # nodes/weights explicitly.
    (
        safe_stage_duration_array,
        upper_slope_array,
        lower_slope_array,
        ub_starts,
        lb_starts,
    ) = _effective_general_schedule(node_array, d, a1, b1_array, a2, b2_array)

    ub_1 = ub_starts[1]
    lb_1 = lb_starts[1]
    half_w_1 = (ub_1 - lb_1) / 2.0
    center_1 = (ub_1 + lb_1) / 2.0
    xs_init = x_ref * half_w_1 + center_1
    ws_init = w_ref * half_w_1

    pv_init = q_single(
        xs_init,
        mu_array[0],
        sigma_array[0],
        a1,
        upper_slope_array[0],
        a2,
        lower_slope_array[0],
        safe_stage_duration_array[0],
        x0,
        trunc_num=trunc_num,
    )

    if log_space:
        ws_pv_init = _to_log_space(ws_init * pv_init)
    else:
        ws_pv_init = ws_init * pv_init

    # Carry tracks the current interface nodes, weighted alive-state mass, and
    # boundary positions at the start of the current stage.
    carry = (xs_init, ws_pv_init, ub_1, lb_1)

    if log_space:

        def stage_step(carry, stage_idx):
            xs_prev, log_ws_pv_prev, ub_prev, lb_prev = carry
            ub_curr = ub_starts[stage_idx + 1]
            lb_curr = lb_starts[stage_idx + 1]
            half_w = (ub_curr - lb_curr) / 2.0
            center = (ub_curr + lb_curr) / 2.0
            xs = x_ref * half_w + center
            ws = w_ref * half_w
            P = q_single(
                xs[:, None],
                mu_array[stage_idx],
                sigma_array[stage_idx],
                ub_prev,
                upper_slope_array[stage_idx],
                lb_prev,
                lower_slope_array[stage_idx],
                safe_stage_duration_array[stage_idx],
                xs_prev[None, :],
                trunc_num=trunc_num,
            )
            log_pv = logsumexp(_to_log_space(P) + log_ws_pv_prev[None, :], axis=1)
            log_ws_pv = _to_log_space(ws) + log_pv
            active = stage_idx < (d - 1)
            xs_out = jnp.where(active, xs, xs_prev)
            ws_pv_out = jnp.where(active, log_ws_pv, log_ws_pv_prev)
            ub_out = jnp.where(active, ub_curr, ub_prev)
            lb_out = jnp.where(active, lb_curr, lb_prev)
            return (xs_out, ws_pv_out, ub_out, lb_out), None

    else:

        def stage_step(carry, stage_idx):
            xs_prev, ws_pv_prev, ub_prev, lb_prev = carry
            ub_curr = ub_starts[stage_idx + 1]
            lb_curr = lb_starts[stage_idx + 1]
            half_w = (ub_curr - lb_curr) / 2.0
            center = (ub_curr + lb_curr) / 2.0
            xs = x_ref * half_w + center
            ws = w_ref * half_w
            P = q_single(
                xs[:, None],
                mu_array[stage_idx],
                sigma_array[stage_idx],
                ub_prev,
                upper_slope_array[stage_idx],
                lb_prev,
                lower_slope_array[stage_idx],
                safe_stage_duration_array[stage_idx],
                xs_prev[None, :],
                trunc_num=trunc_num,
            )
            pv = P @ ws_pv_prev
            ws_pv = ws * pv
            active = stage_idx < (d - 1)
            xs_out = jnp.where(active, xs, xs_prev)
            ws_pv_out = jnp.where(active, ws_pv, ws_pv_prev)
            ub_out = jnp.where(active, ub_curr, ub_prev)
            lb_out = jnp.where(active, lb_curr, lb_prev)
            return (xs_out, ws_pv_out, ub_out, lb_out), None

    if max_d > 2:
        carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 1))

    # Evaluate the observed hit density in the true final stage and reduce over
    # the latent start-position quadrature grid.
    xs_final, ws_pv_final, _, _ = carry
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)
    ub_final = ub_starts[safe_d_idx]
    lb_final = lb_starts[safe_d_idx]
    node_final = node_array[safe_d_idx]
    mu_final = mu_array[safe_d_idx]
    sigma_final = sigma_array[safe_d_idx]
    b1_final = b1_array[safe_d_idx]
    b2_final = b2_array[safe_d_idx]
    t_in_final_stage = rt - node_final

    fptds = fptd_single(
        t_in_final_stage,
        mu_final,
        sigma_final,
        ub_final,
        b1_final,
        lb_final,
        b2_final,
        xs_final,
        choice,
        trunc_num=trunc_num,
    )

    if log_space:
        return jnp.exp(logsumexp(_to_log_space(fptds) + ws_pv_final))
    return jnp.sum(fptds * ws_pv_final)


def compute_heterog_multistage_fptd_stagescan(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Public single-trial generalized multistage stage-scan wrapper.

    This is the public stage-scan counterpart to
    :func:`compute_heterog_multistage_fptd_precomputed`. The default alias
    :func:`compute_heterog_multistage_fptd` still points to the precomputed
    production path.
    """

    def single_fn(_):
        return fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _heterog_multistage_fptd_stagescan(
            rt,
            choice,
            x0,
            a1,
            a2,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            d,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


# ---------------------------------------------------------------------------
# Public aliases and JIT wrappers
# ---------------------------------------------------------------------------


compute_addm_fptd_precomputed_jit = jit(
    compute_addm_fptd_precomputed,
    static_argnames=("order", "trunc_num", "log_space"),
)
compute_addm_fptd_stagescan_jit = jit(
    compute_addm_fptd_stagescan,
    static_argnames=("order", "trunc_num", "log_space"),
)
compute_heterog_multistage_fptd_precomputed_jit = jit(
    compute_heterog_multistage_fptd_precomputed,
    static_argnames=("order", "trunc_num", "log_space"),
)
compute_heterog_multistage_fptd_stagescan_jit = jit(
    compute_heterog_multistage_fptd_stagescan,
    static_argnames=("order", "trunc_num", "log_space"),
)

# Plain names keep pointing to the production precomputed kernels.
compute_addm_fptd = compute_addm_fptd_precomputed
compute_addm_fptd_jit = compute_addm_fptd_precomputed_jit
compute_heterog_multistage_fptd = compute_heterog_multistage_fptd_precomputed
compute_heterog_multistage_fptd_jit = compute_heterog_multistage_fptd_precomputed_jit
