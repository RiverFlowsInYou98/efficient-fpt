"""Batched aDDM likelihood and NLL computation using JAX kernels.

This module contains the JAX batch layer for aDDM likelihoods and NLLs. It
exposes two explicit batch methods:

- a legacy baseline that vmaps the production single-trial kernel in
  ``jax.multi_stage`` over trials
- a dedicated batched stage-scan kernel that carries the whole batch through
  one multistage recurrence

These batch variants now sit alongside the single-trial variants in
``jax.multi_stage``:

- ``jax.multi_stage.compute_*_precomputed``: public single-trial production
  kernels
- ``jax.multi_stage.compute_*_stagescan``: public single-trial stage-scan
  kernels
- ``jax.batch.compute_addm_likelihoods_batchvmap``: legacy batch baseline
- ``jax.batch.compute_addm_likelihoods_batchscan``: production batch kernel
- ``jax.batch.compute_addm_likelihoods``: alias to
  ``compute_addm_likelihoods_batchscan``
- ``jax.batch.make_addm_nll_function_batchvmap``: legacy optimizer closure
- ``jax.batch.make_addm_nll_function_batchscan``: production optimizer closure
- ``jax.batch.make_addm_nll_function``: alias to
  ``make_addm_nll_function_batchscan``

There is currently no batched kernel that precomputes *all* stage transition
matrices for the general current batch API up front. That design is possible
in principle, but because stage drifts and stage schedules vary by trial, the
fully general transition tensor would scale roughly like
``(batch_size, max_d - 2, order, order)``.

JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
``adaptive_stopping`` or ``threshold`` option in this backend.
"""

import warnings

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, remat, vmap
from jax.scipy.special import logsumexp

from .addm_helpers import _build_addm_mu_array_data
from .multi_stage import _to_log_space, compute_addm_fptd
from .single_stage import fptd_single, q_single
from .utils import get_gauss_legendre_ref, _DUMMY_STAGE_DURATION
from .._defaults import DEFAULT_QUADRATURE_ORDER, DEFAULT_TRUNC_NUM


# ---------------------------------------------------------------------------
# Warning and NLL reduction helpers
# ---------------------------------------------------------------------------


def _warn_invalid_likelihoods(likelihoods):
    """Emit deterministic warnings for skipped trial likelihoods.

    Parameters
    ----------
    likelihoods : array-like, shape (n_trials,)
        Per-trial likelihood values. Exact zeros and invalid values trigger the
        same warning messages used by the other backends.
    """
    likelihoods_np = np.ravel(np.asarray(likelihoods))
    for idx, value in enumerate(likelihoods_np):
        if value == 0.0:
            warnings.warn(
                f"trial {idx} outputs 0 likelihood, skipped",
                RuntimeWarning,
                stacklevel=2,
            )
        elif not np.isfinite(value) or value < 0.0:
            warnings.warn(
                f"trial {idx} outputs invalid likelihood, skipped",
                RuntimeWarning,
                stacklevel=2,
            )


def _reduce_addm_likelihoods_to_nll(likelihoods, reduce="mean", warn=True):
    """Reduce a likelihood vector to a mean or summed negative log-likelihood.

    Invalid trials are skipped using the shared project rule:

    - valid if finite and strictly positive
    - skipped if zero, negative, or non-finite
    - return ``NaN`` if every trial is skipped

    Parameters
    ----------
    likelihoods : jax.Array, shape (n_trials,)
        Per-trial likelihood values.
    reduce : {"mean", "sum"}, optional
        Reduction to apply after filtering invalid trials.
    warn : bool, optional
        Whether to emit deterministic Python warnings for skipped trials.
    """
    valid = jnp.isfinite(likelihoods) & (likelihoods > 0)
    if warn:
        _warn_invalid_likelihoods(likelihoods)

    safe_likelihoods = jnp.where(valid, likelihoods, 1.0)
    losses = jnp.where(valid, -jnp.log(safe_likelihoods), 0.0)
    total_loss = jnp.sum(losses)
    num_valid = jnp.sum(valid.astype(jnp.int32))
    if reduce == "sum":
        return jnp.where(num_valid > 0, total_loss, jnp.nan)
    return jnp.where(num_valid > 0, total_loss / num_valid, jnp.nan)


# ---------------------------------------------------------------------------
# Batched schedule builders
# ---------------------------------------------------------------------------


def _safe_stage_durations_batch(sacc_array_data, d_data):
    """Build numerically safe stage durations for a padded batch of trials.

    Parameters
    ----------
    sacc_array_data : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for a batch of aDDM trials.
    d_data : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.

    Returns
    -------
    valid_stage_mask_data : jax.Array, shape (n_trials, max_d - 1)
        Boolean mask marking which entries of ``diff(sacc_array_data, axis=1)``
        correspond to real stage-to-stage durations.
    safe_stage_duration_array_data : jax.Array, shape (n_trials, max_d - 1)
        Per-transition duration matrix used by the batched JAX kernels. Valid
        durations are passed through unchanged. Padded transitions are replaced
        by ``_DUMMY_STAGE_DURATION`` so traced JAX code never sees padded zero
        or negative durations from the tail.

    Notes
    -----
    This is the batched analogue of ``jax.multi_stage._safe_stage_durations``.

    Example
    -------
    If

    ``sacc_array_data = [[0, 1, 3, 7, 0], [0, 0.5, 0, 0, 0]]``

    and

    ``d_data = [4, 2]``,

    then the real durations are ``[1, 2, 4]`` for the first trial and
    ``[0.5]`` for the second trial, giving

    ``valid_stage_mask_data = [[True, True, True, False], [True, False, False, False]]``

    and

    ``safe_stage_duration_array_data = [[1, 2, 4, dummy], [0.5, dummy, dummy, dummy]]``.
    """
    batch_size, max_d = sacc_array_data.shape
    dtype = sacc_array_data.dtype
    if max_d <= 1:
        return (
            jnp.empty((batch_size, 0), dtype=bool),
            jnp.empty((batch_size, 0), dtype=dtype),
        )

    raw_stage_durations = jnp.diff(sacc_array_data, axis=1)
    stage_idx = jnp.arange(max_d - 1)[None, :]
    valid_stage_mask_data = stage_idx < (d_data[:, None] - 1)
    safe_stage_duration_array_data = jnp.where(
        valid_stage_mask_data,
        raw_stage_durations,
        _DUMMY_STAGE_DURATION,
    )
    return valid_stage_mask_data, safe_stage_duration_array_data


def _effective_addm_batch_schedule(sacc_array_data, d_data, a, b):
    """Construct the batched aDDM boundary schedule.

    Parameters
    ----------
    sacc_array_data : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for each trial.
    d_data : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.
    a : float
        Initial upper-boundary intercept. The lower boundary starts at ``-a``.
    b : float
        Symmetric boundary-collapse slope magnitude. The upper boundary slope is
        ``-b`` and the lower boundary slope is ``+b`` on active stages.

    Returns
    -------
    safe_stage_duration_array_data : jax.Array, shape (n_trials, max_d - 1)
        Safe per-stage durations from :func:`_safe_stage_durations_batch`.
    upper_slope_array_data : jax.Array, shape (n_trials, max_d - 1)
        Upper-boundary slope for each transition. Active entries are ``-b`` and
        padded entries are ``0``.
    lower_slope_array_data : jax.Array, shape (n_trials, max_d - 1)
        Lower-boundary slope for each transition. Active entries are ``+b`` and
        padded entries are ``0``.
    a_starts_data : jax.Array, shape (n_trials, max_d)
        Upper-boundary value at the start of each stage.

    Notes
    -----
    This is the batched analogue of
    ``jax.multi_stage._effective_addm_schedule``. The padded tail is made
    inert by combining dummy positive durations with zero slopes.
    """
    batch_size, max_d = sacc_array_data.shape
    dtype = sacc_array_data.dtype
    if max_d <= 1:
        empty = jnp.empty((batch_size, 0), dtype=dtype)
        starts = jnp.full((batch_size, 1), a, dtype=dtype)
        return empty, empty, empty, starts

    valid_stage_mask_data, safe_stage_duration_array_data = _safe_stage_durations_batch(
        sacc_array_data, d_data
    )
    upper_slope_array_data = jnp.where(valid_stage_mask_data, -b, 0.0)
    lower_slope_array_data = jnp.where(valid_stage_mask_data, b, 0.0)
    a_starts_data = jnp.concatenate(
        [
            jnp.full((batch_size, 1), a, dtype=dtype),
            a
            + jnp.cumsum(
                upper_slope_array_data * safe_stage_duration_array_data, axis=1
            ),
        ],
        axis=1,
    )
    return (
        safe_stage_duration_array_data,
        upper_slope_array_data,
        lower_slope_array_data,
        a_starts_data,
    )


# ---------------------------------------------------------------------------
# Batch likelihood kernels
# ---------------------------------------------------------------------------


def compute_addm_likelihoods_batchvmap(
    rt_data,
    choice_data,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Legacy batch baseline: vmap over the scalar ADDM kernel.

    Notes
    -----
    This is the simplest batch implementation: it vmaps the public
    single-trial production kernel ``compute_addm_fptd`` over trials. Since the
    single-trial production kernel in ``jax.multi_stage`` already precomputes
    stage transition matrices per trial, this path should be thought of as
    "batch by vmapping the single-trial precomputed kernel".

    Within the current JAX organization, this function is the explicit legacy
    batch baseline. The explicit single-trial stage-scan kernels live in
    ``jax.multi_stage`` under ``compute_*_stagescan``.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is
    no ``adaptive_stopping`` or ``threshold`` option in this backend.

    If ``use_remat=True``, rematerialize the vmapped single-trial production
    kernel to trade extra compute for lower reverse-mode memory use.
    """

    def single_trial_likelihood(rt, choice, r1, r2, flag, sacc_array, d):
        return compute_addm_fptd(
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
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    if use_remat:
        single_trial_likelihood = remat(single_trial_likelihood)

    return vmap(single_trial_likelihood, in_axes=(0, 0, 0, 0, 0, 0, 0))(
        rt_data, choice_data, r1_data, r2_data, flag_data, sacc_array_data, d_data
    )


def _compute_addm_likelihoods_batchscan_core(
    rt_data,
    choice_data,
    mu_array_data,
    sacc_array_data,
    d_data,
    sigma,
    a,
    b,
    x0,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Compute batched aDDM likelihoods with a dedicated stage-scan kernel.

    Parameters
    ----------
    rt_data, choice_data : jax.Array, shape (n_trials,)
        Observed reaction times and choices.
    mu_array_data : jax.Array, shape (n_trials, max_d)
        Per-trial, per-stage drift arrays already derived from the aDDM
        covariates.
    sacc_array_data : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for each trial.
    d_data : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.
    sigma, a, b, x0 : float
        Shared aDDM parameters for the batch.
    order, trunc_num : int, optional
        Quadrature order and fixed single-stage truncation length.
    log_space : bool, optional
        Whether to propagate the alive-state mass in log space.
    use_remat : bool, optional
        If True, rematerialize the scan body to trade extra compute for lower
        reverse-mode memory use.

    Returns
    -------
    jax.Array, shape (n_trials,)
        Per-trial likelihoods.

    Notes
    -----
    This is the main batched runtime kernel. It replaces the older
    ``vmap(compute_addm_fptd)`` baseline with a dedicated stage-wise batched
    scan that carries all trials together through the same quadrature updates.

    Unlike the single-trial production kernel in ``jax.multi_stage``, this
    function does *not* precompute all transition matrices for all stages and
    all trials up front. Instead, it computes each stage's batched transition
    matrix inside the scan body. That choice keeps memory usage and XLA temp
    storage lower than a hypothetical fully precomputed batch-transition
    tensor.

    So within the JAX backend, the four main internal variants are:

    - ``jax.multi_stage.compute_*_precomputed``: single-trial production,
      precompute all stage transitions for one trial
    - ``jax.multi_stage.compute_*_stagescan``: single-trial stage-scan,
      compute transitions inside ``lax.scan``
    - ``compute_addm_likelihoods_batchvmap``: legacy batch baseline, vmap the
      single-trial production path over trials
    - ``compute_addm_likelihoods_batchscan``: production batch kernel, compute
      batched transitions on the fly inside one shared scan
    """
    # Reference Gauss-Legendre quadrature on [-1, 1].
    x_ref, w_ref = get_gauss_legendre_ref(order)

    batch_size, max_d = mu_array_data.shape
    # Handle d == 1 trials directly through the single-stage kernel so the
    # multistage recurrence only needs to serve the true multistage cases.
    upper_single = fptd_single(
        rt_data,
        mu_array_data[:, 0],
        sigma,
        a,
        -b,
        -a,
        b,
        x0,
        1,
        trunc_num=trunc_num,
    )
    lower_single = fptd_single(
        rt_data,
        mu_array_data[:, 0],
        sigma,
        a,
        -b,
        -a,
        b,
        x0,
        -1,
        trunc_num=trunc_num,
    )
    single_stage = jnp.where(choice_data == 1, upper_single, lower_single)
    if max_d < 2:
        return single_stage

    # Build per-trial stage schedules and place the first interface quadrature
    # grid. xs_init / ws_init have shape (n_trials, order).
    (
        safe_stage_duration_array_data,
        upper_slope_array_data,
        lower_slope_array_data,
        a_starts_data,
    ) = _effective_addm_batch_schedule(sacc_array_data, d_data, a, b)

    a_1 = a_starts_data[:, 1]
    xs_init = x_ref[None, :] * a_1[:, None]
    ws_init = w_ref[None, :] * a_1[:, None]

    # Propagate x0 through stage 0 for every trial to reach the first
    # interface quadrature grid.
    pv_init = vmap(
        lambda xs, mu, upper_slope, lower_slope, dt: q_single(
            xs,
            mu,
            sigma,
            a,
            upper_slope,
            -a,
            lower_slope,
            dt,
            x0,
            trunc_num=trunc_num,
        )
    )(
        xs_init,
        mu_array_data[:, 0],
        upper_slope_array_data[:, 0],
        lower_slope_array_data[:, 0],
        safe_stage_duration_array_data[:, 0],
    )

    if log_space:
        # Carry = (current interface nodes, log(weighted alive-state mass)).
        carry = (xs_init, _to_log_space(ws_init * pv_init))

        def stage_step(carry, stage_idx):
            xs_prev, log_ws_pv_prev = carry

            # Build the next interface grid for every trial, then compute the
            # stage transition matrix P for each trial on the fly.
            a_prev = a_starts_data[:, stage_idx]
            a_curr = a_starts_data[:, stage_idx + 1]
            xs = x_ref[None, :] * a_curr[:, None]
            ws = w_ref[None, :] * a_curr[:, None]

            P = vmap(
                lambda xs_row, mu, a_prev_val, upper_slope, lower_slope, dt, xs_prev_row: q_single(
                    xs_row[:, None],
                    mu,
                    sigma,
                    a_prev_val,
                    upper_slope,
                    -a_prev_val,
                    lower_slope,
                    dt,
                    xs_prev_row[None, :],
                    trunc_num=trunc_num,
                )
            )(
                xs,
                mu_array_data[:, stage_idx],
                a_prev,
                upper_slope_array_data[:, stage_idx],
                lower_slope_array_data[:, stage_idx],
                safe_stage_duration_array_data[:, stage_idx],
                xs_prev,
            )

            log_pv_new = logsumexp(
                _to_log_space(P) + log_ws_pv_prev[:, None, :], axis=2
            )
            log_ws_pv_new = _to_log_space(ws) + log_pv_new
            active = stage_idx < (d_data - 1)

            xs_out = jnp.where(active[:, None], xs, xs_prev)
            log_ws_pv_out = jnp.where(active[:, None], log_ws_pv_new, log_ws_pv_prev)
            return (xs_out, log_ws_pv_out), None

        if use_remat:
            stage_step = remat(stage_step)

        if max_d > 2:
            carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 1))

        xs_final, log_ws_pv_final = carry
    else:
        # Carry = (current interface nodes, weighted alive-state mass).
        carry = (xs_init, ws_init * pv_init)

        def stage_step(carry, stage_idx):
            xs_prev, ws_pv_prev = carry

            # Build the next interface grid for every trial, then compute the
            # stage transition matrix P for each trial on the fly.
            a_prev = a_starts_data[:, stage_idx]
            a_curr = a_starts_data[:, stage_idx + 1]
            xs = x_ref[None, :] * a_curr[:, None]
            ws = w_ref[None, :] * a_curr[:, None]

            P = vmap(
                lambda xs_row, mu, a_prev_val, upper_slope, lower_slope, dt, xs_prev_row: q_single(
                    xs_row[:, None],
                    mu,
                    sigma,
                    a_prev_val,
                    upper_slope,
                    -a_prev_val,
                    lower_slope,
                    dt,
                    xs_prev_row[None, :],
                    trunc_num=trunc_num,
                )
            )(
                xs,
                mu_array_data[:, stage_idx],
                a_prev,
                upper_slope_array_data[:, stage_idx],
                lower_slope_array_data[:, stage_idx],
                safe_stage_duration_array_data[:, stage_idx],
                xs_prev,
            )

            pv_new = jnp.matmul(P, ws_pv_prev[..., None]).squeeze(axis=-1)
            ws_pv_new = ws * pv_new
            active = stage_idx < (d_data - 1)

            xs_out = jnp.where(active[:, None], xs, xs_prev)
            ws_pv_out = jnp.where(active[:, None], ws_pv_new, ws_pv_prev)
            return (xs_out, ws_pv_out), None

        if use_remat:
            stage_step = remat(stage_step)

        if max_d > 2:
            carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 1))

        xs_final, ws_pv_final = carry

    # Pick the true final stage for each trial, evaluate the observed
    # boundary-hit density there, then reduce over the final-stage latent
    # start-position quadrature grid.
    safe_d_idx = jnp.minimum(d_data - 1, max_d - 1)
    sacc_final = jnp.take_along_axis(
        sacc_array_data, safe_d_idx[:, None], axis=1
    ).squeeze(axis=1)
    a_final = jnp.take_along_axis(a_starts_data, safe_d_idx[:, None], axis=1).squeeze(
        axis=1
    )
    mu_final = jnp.take_along_axis(mu_array_data, safe_d_idx[:, None], axis=1).squeeze(
        axis=1
    )
    t_in_final_stage = rt_data - sacc_final

    upper = fptd_single(
        t_in_final_stage[:, None],
        mu_final[:, None],
        sigma,
        a_final[:, None],
        -b,
        -a_final[:, None],
        b,
        xs_final,
        1,
        trunc_num=trunc_num,
    )
    lower = fptd_single(
        t_in_final_stage[:, None],
        mu_final[:, None],
        sigma,
        a_final[:, None],
        -b,
        -a_final[:, None],
        b,
        xs_final,
        -1,
        trunc_num=trunc_num,
    )
    fptds = jnp.where(choice_data[:, None] == 1, upper, lower)

    if log_space:
        multi_stage = jnp.exp(logsumexp(_to_log_space(fptds) + log_ws_pv_final, axis=1))
    else:
        multi_stage = jnp.sum(fptds * ws_pv_final, axis=1)

    return jnp.where(d_data == 1, single_stage, multi_stage)


def compute_addm_likelihoods_batchscan(
    rt_data,
    choice_data,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Compute ADDM likelihoods with the dedicated batch stage-scan kernel.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is
    no ``adaptive_stopping`` or ``threshold`` option in this backend.

    This public wrapper keeps the high-level addm signature
    ``(eta, kappa, r1_data, r2_data, flag_data)`` and dispatches to the
    explicit batch stage-scan kernel after building the derived drifts.

    If ``use_remat=True``, rematerialize the batch scan body to trade extra
    compute for lower reverse-mode memory use.
    """
    mu_array_data = _build_addm_mu_array_data(
        eta, kappa, r1_data, r2_data, flag_data, d_data, sacc_array_data.shape[1]
    )
    return _compute_addm_likelihoods_batchscan_core(
        rt_data,
        choice_data,
        mu_array_data,
        sacc_array_data,
        d_data,
        sigma,
        a,
        b,
        x0,
        order=order,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )


# ---------------------------------------------------------------------------
# Public batch reductions
# ---------------------------------------------------------------------------


def compute_addm_nll(
    rt_data,
    choice_data,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
    reduce="mean",
    warn=True,
):
    """Compute negative log-likelihood for a batch of addm trials.

    Parameters
    ----------
    order, trunc_num : int, optional
        Quadrature order and fixed single-stage truncation length. JAX does not
        expose ``adaptive_stopping`` or ``threshold`` in this backend.
    use_remat : bool, optional
        If True, rematerialize the selected batch likelihood kernel to trade
        extra compute for lower reverse-mode memory use.
    reduce : str, optional
        ``"mean"`` (default) or ``"sum"``.
    warn : bool, optional
        If True, emit warnings for skipped trials.
    """
    likelihoods = compute_addm_likelihoods(
        rt_data,
        choice_data,
        eta,
        kappa,
        sigma,
        a,
        b,
        x0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order=order,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )
    return _reduce_addm_likelihoods_to_nll(likelihoods, reduce=reduce, warn=warn)


# ---------------------------------------------------------------------------
# NLL closure builders
# ---------------------------------------------------------------------------


def _build_addm_nll_function_with_kernel(
    kernel,
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Build a jitted parameter-only NLL closure from a likelihood kernel.

    The returned function closes over fixed data and exposes only the model
    parameters ``(eta, kappa, sigma, a, b, x0)``.
    """

    @jit
    def nll_fn(eta, kappa, sigma, a, b, x0):
        likelihoods = kernel(
            rt_data,
            choice_data,
            eta,
            kappa,
            sigma,
            a,
            b,
            x0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
            use_remat=use_remat,
        )
        return _reduce_addm_likelihoods_to_nll(likelihoods, reduce="sum", warn=False)

    return nll_fn


def make_addm_nll_function_batchvmap(
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Build an optimization closure using the legacy vmap baseline kernel.

    If ``use_remat=True``, rematerialize the vmapped single-trial production
    kernel to trade extra compute for lower reverse-mode memory use.
    """

    def kernel(
        rt_data,
        choice_data,
        eta,
        kappa,
        sigma,
        a,
        b,
        x0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        *,
        order=DEFAULT_QUADRATURE_ORDER,
        trunc_num=DEFAULT_TRUNC_NUM,
        log_space=False,
        use_remat=False,
    ):
        return compute_addm_likelihoods_batchvmap(
            rt_data,
            choice_data,
            eta,
            kappa,
            sigma,
            a,
            b,
            x0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
            use_remat=use_remat,
        )

    return _build_addm_nll_function_with_kernel(
        kernel,
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order=order,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )


def make_addm_nll_function_batchscan(
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order=DEFAULT_QUADRATURE_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Create a batched ADDM negative log-likelihood function for optimization.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is
    no ``adaptive_stopping`` or ``threshold`` option in this backend.

    If ``use_remat=True``, rematerialize the batch stage-scan body to trade
    extra compute for lower reverse-mode memory use.
    """

    def kernel(
        rt_data,
        choice_data,
        eta,
        kappa,
        sigma,
        a,
        b,
        x0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        *,
        order=order,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    ):
        mu_array_data = _build_addm_mu_array_data(
            eta, kappa, r1_data, r2_data, flag_data, d_data, sacc_array_data.shape[1]
        )
        return _compute_addm_likelihoods_batchscan_core(
            rt_data,
            choice_data,
            mu_array_data,
            sacc_array_data,
            d_data,
            sigma,
            a,
            b,
            x0,
            order=order,
            trunc_num=trunc_num,
            log_space=log_space,
            use_remat=use_remat,
        )

    return _build_addm_nll_function_with_kernel(
        kernel,
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order=order,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )

# ---------------------------------------------------------------------------
# Public aliases and JIT wrappers
# ---------------------------------------------------------------------------


compute_addm_likelihoods = compute_addm_likelihoods_batchscan
make_addm_nll_function = make_addm_nll_function_batchscan
compute_addm_likelihoods_jit = jit(
    compute_addm_likelihoods,
    static_argnames=("order", "trunc_num", "log_space", "use_remat"),
)
