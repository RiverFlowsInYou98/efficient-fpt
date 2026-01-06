"""
JAX implementation of multi-stage first-passage time density computation.

Uses jax.lax.scan for efficient sequential stage propagation and
broadcasting for transition matrix computation.
"""

import jax.numpy as jnp
from jax import lax, jit

from .single_stage import fptd_single_jax, q_single_jax
from .utils import GAUSS_LEGENDRE_30_X, GAUSS_LEGENDRE_30_W


def pad_sacc_array_safely(sacc_array, d, max_d):
    """
    Create safe padding for saccade array to avoid NaN during gradient computation.
    
    The Problem:
    When padding sacc_array with zeros (e.g., [0.0, 0.8, 1.6, 0.0, 0.0] for d=3),
    the invalid stages compute T_curr = 0.0 - 0.0 = 0, causing division by zero.
    Even though results are masked out, JAX evaluates both branches during
    gradient computation, propagating NaN into the gradients.
    
    The Solution:
    Pad with increasing times so T_curr is always positive. The invalid stages
    compute valid (but ignored) numbers, so gradients remain finite.
    
    Parameters
    ----------
    sacc_array : array of shape (max_d,)
        Saccade times with zero-padding for unused stages
    d : int
        Actual number of stages
    max_d : int
        Maximum stages (array length)
        
    Returns
    -------
    safe_sacc : array of shape (max_d,)
        Saccade times with safe padding (increasing times)
        
    Example
    -------
    >>> sacc = [0.0, 0.8, 1.6, 0.0, 0.0]  # d=3, padded to max_d=5
    >>> safe_sacc = pad_sacc_array_safely(sacc, d=3, max_d=5)
    >>> # safe_sacc ≈ [0.0, 0.8, 1.6, 2.4, 3.2]
    >>> # Now T_curr = 2.4 - 1.6 = 0.8 > 0 for padded stages!
    """
    # Compute average interval from valid stages
    last_valid_sacc = sacc_array[d - 1]
    avg_interval = jnp.where(d > 1, last_valid_sacc / (d - 1), 1.0)
    
    # Create safe padding: extend with increasing times
    indices = jnp.arange(max_d)
    # For indices >= d, use: last_valid_sacc + avg_interval * (idx - d + 1)
    safe_extension = last_valid_sacc + avg_interval * (indices - d + 1)
    
    # Keep original values for valid indices, use safe extension otherwise
    safe_sacc = jnp.where(indices < d, sacc_array, safe_extension)
    
    return safe_sacc


def get_addm_fptd_jax(t, d, mu_array, sacc_array, sigma, a, b, x0, bdy, 
                       order=30, trunc_num=50):
    """
    Multi-stage FPTD for attention-dependent drift diffusion model.
    
    Computes the first passage time density for a process with piecewise
    constant drift rates and symmetric collapsing boundaries.
    
    This version uses SAFE PADDING to support both:
    - vmap over trials with varying d
    - Automatic differentiation (gradients)
    
    Parameters
    ----------
    t : float
        First passage time (reaction time)
    d : int
        Number of stages (must be >= 1). Can be traced for vmap.
    mu_array : array of shape (max_d,)
        Drift rates for each stage. Only first d elements are used.
        Padded elements can be any value (they are masked out).
    sacc_array : array of shape (max_d,)
        Stage start times. sacc_array[0] should be 0.
        Padded elements can be zeros (safe padding is applied internally).
    sigma : float
        Diffusion coefficient
    a : float
        Initial boundary separation (symmetric: upper=a, lower=-a)
    b : float
        Boundary collapse rate (upper: a-b*t, lower: -a+b*t)
    x0 : float
        Starting position
    bdy : int
        Boundary indicator: 1 for upper, -1 for lower
    order : int
        Quadrature order (must be 30 currently)
    trunc_num : int
        Number of series terms for density computation
        
    Returns
    -------
    density : float
        First passage time density at time t for boundary bdy
        
    Notes
    -----
    The boundaries are:
    - Upper: u(t) = a - b*t
    - Lower: l(t) = -a + b*t
    
    Safe Padding for Gradients:
    This function internally applies safe padding to sacc_array to ensure
    that padded stages compute valid (non-NaN) intermediate values. This
    allows gradients to flow correctly even when using jnp.where masking.
    
    See `pad_sacc_array_safely` for details on how this works.
    """
    # Get quadrature nodes/weights on reference interval [-1, 1]
    if order != 30:
        raise ValueError("Currently only order=30 is supported")
    x_ref = GAUSS_LEGENDRE_30_X
    w_ref = GAUSS_LEGENDRE_30_W
    
    # Apply safe padding to sacc_array
    max_d = mu_array.shape[0]
    safe_sacc = pad_sacc_array_safely(sacc_array, d, max_d)
    
    # Single-stage result (always computed)
    single_stage_result = fptd_single_jax(
        t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num
    )
    
    # Multi-stage case: propagate distribution through stages
    # Initialize from stage 0 to stage 1
    a_1 = a - b * safe_sacc[1]  # Boundary at end of stage 0
    xs_init = x_ref * a_1       # Quadrature points scaled to [-a_1, a_1]
    ws_init = w_ref * a_1       # Quadrature weights scaled
    
    # Compute distribution at end of stage 0
    pv_init = q_single_jax(
        xs_init,  # (order,) destinations
        mu_array[0], sigma, 
        a, -b, -a, b,  # Boundaries at t=0
        safe_sacc[1], # Duration of stage 0
        x0,            # Scalar source
        trunc_num
    )
    
    init_carry = (xs_init, ws_init * pv_init, a_1, safe_sacc[1], mu_array[1])
    
    def stage_step(carry, stage_idx):
        """Propagate distribution through one stage with masking."""
        xs_prev, ws_pv_prev, a_prev, sacc_prev, mu_final_prev = carry
        
        # Current stage parameters
        mu = mu_array[stage_idx]
        safe_idx = jnp.minimum(stage_idx + 1, max_d - 1)
        sacc_curr = safe_sacc[safe_idx]
        # T_curr is now guaranteed positive due to safe padding!
        T_curr = sacc_curr - sacc_prev
        a_curr = a - b * sacc_curr
        
        # New quadrature points for end of this stage
        xs = x_ref * a_curr
        ws = w_ref * a_curr
        
        # Compute transition matrix using BROADCASTING
        P = q_single_jax(
            xs[:, None],           # (order, 1) destinations
            mu, sigma, 
            a_prev, -b, -a_prev, b, # Boundaries at stage start
            T_curr,                 # Stage duration
            xs_prev[None, :],       # (1, order) sources
            trunc_num
        )
        
        # Weighted sum: pv[i] = sum_j P[i,j] * ws_pv_prev[j]
        pv = jnp.sum(ws_pv_prev * P, axis=1)
        ws_pv = ws * pv
        
        # Mask invalid stages (results are valid numbers, just ignored)
        valid = stage_idx < d - 1
        xs_out = jnp.where(valid, xs, xs_prev)
        ws_pv_out = jnp.where(valid, ws_pv, ws_pv_prev)
        a_out = jnp.where(valid, a_curr, a_prev)
        sacc_out = jnp.where(valid, sacc_curr, sacc_prev)
        # Track final stage drift
        is_final = stage_idx == d - 2
        next_mu = mu_array[jnp.minimum(stage_idx + 1, max_d - 1)]
        mu_final_out = jnp.where(is_final, next_mu, mu_final_prev)
        
        return (xs_out, ws_pv_out, a_out, sacc_out, mu_final_out), None
    
    # Scan over all possible stages
    (xs_final, ws_pv_final, a_final, sacc_final, mu_final), _ = lax.scan(
        stage_step,
        init_carry,
        jnp.arange(1, max_d - 1)
    )
    
    # Final stage: compute FPTD
    t_in_final_stage = t - sacc_final
    
    fptds = fptd_single_jax(
        t_in_final_stage,      # Time in final stage
        mu_final,              # Drift in final stage
        sigma, 
        a_final, -b, -a_final, b,
        xs_final,
        bdy, 
        trunc_num
    )
    
    multi_stage_result = jnp.sum(fptds * ws_pv_final)
    
    # Select between single-stage and multi-stage
    return jnp.where(d == 1, single_stage_result, multi_stage_result)


# JIT-compiled version with static bdy and order
get_addm_fptd_jax_jit = jit(get_addm_fptd_jax, static_argnums=(8, 9, 10))

