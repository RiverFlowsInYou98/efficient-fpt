"""
Batch computation of likelihoods using JAX vmap.

Provides efficient batch evaluation of first-passage time densities
over multiple trials, with an optional memory-efficient chunked path
using ``lax.scan`` + ``jax.checkpoint`` for large datasets on GPU.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
from functools import partial

from .multi_stage import get_addm_fptd_jax, get_addm_fptd_jax_fast


def compute_likelihoods_batch(
    rt_data,          # (n_trials,)
    choice_data,      # (n_trials,)
    mu_array_data,    # (n_trials, max_d)
    sacc_array_data,  # (n_trials, max_d)
    length_data,      # (n_trials,)
    sigma, a, b, x0,
    order=30, 
    trunc_num=50
):
    """
    Compute likelihoods for a batch of trials.
    
    All trial data must be padded to the same max_d for batching.
    
    Parameters
    ----------
    rt_data : array of shape (n_trials,)
        Reaction times
    choice_data : array of shape (n_trials,)
        Choices (1 for upper boundary, -1 for lower)
    mu_array_data : array of shape (n_trials, max_d)
        Drift rates for each trial, padded to max_d
    sacc_array_data : array of shape (n_trials, max_d)
        Stage start times for each trial, padded to max_d
    length_data : array of shape (n_trials,)
        Number of actual stages for each trial
    sigma : float
        Diffusion coefficient (same for all trials)
    a : float
        Initial boundary separation (same for all trials)
    b : float
        Boundary collapse rate (same for all trials)
    x0 : float
        Starting position (same for all trials)
    order : int
        Quadrature order
    trunc_num : int
        Number of series terms
        
    Returns
    -------
    likelihoods : array of shape (n_trials,)
        Likelihood values for each trial
    """
    def single_trial_likelihood(rt, choice, mu_array, sacc_array, d):
        return get_addm_fptd_jax(
            rt, d, mu_array, sacc_array, sigma, a, b, x0, choice,
            order, trunc_num
        )
    
    # vmap over all trial-specific inputs
    batched_fn = vmap(single_trial_likelihood, in_axes=(0, 0, 0, 0, 0))
    
    return batched_fn(rt_data, choice_data, mu_array_data, sacc_array_data, length_data)


def compute_nll_batch(likelihoods, eps=1e-10):
    """
    Compute mean negative log-likelihood from likelihood values.
    
    Handles zero/negative likelihoods gracefully by clamping.
    
    Parameters
    ----------
    likelihoods : array
        Likelihood values (should be positive)
    eps : float
        Small value to clamp likelihoods to avoid log(0)
        
    Returns
    -------
    nll : float
        Mean negative log-likelihood
    """
    # Clamp to avoid log(0)
    safe_likelihoods = jnp.maximum(likelihoods, eps)
    return -jnp.mean(jnp.log(safe_likelihoods))


def compute_nll_batch_sum(likelihoods, eps=1e-10):
    """
    Compute sum of negative log-likelihoods.
    
    Useful for optimization where we want the total NLL.
    
    Parameters
    ----------
    likelihoods : array
        Likelihood values (should be positive)
    eps : float
        Small value to clamp likelihoods to avoid log(0)
        
    Returns
    -------
    nll_sum : float
        Sum of negative log-likelihoods
    """
    safe_likelihoods = jnp.maximum(likelihoods, eps)
    return -jnp.sum(jnp.log(safe_likelihoods))


def make_nll_function(rt_data, choice_data, mu_array_data, sacc_array_data, 
                      length_data, x0=0.0, order=30, trunc_num=50):
    """
    Create a negative log-likelihood function for optimization.
    
    Returns a function that takes (sigma, a, b) and returns the NLL.
    This is useful for gradient-based optimization.
    
    Parameters
    ----------
    rt_data, choice_data, mu_array_data, sacc_array_data, length_data :
        Trial data (see compute_likelihoods_batch)
    x0 : float
        Starting position
    order : int
        Quadrature order
    trunc_num : int
        Number of series terms
        
    Returns
    -------
    nll_fn : callable
        Function that takes params array [sigma, a, b] and returns NLL
        
    Example
    -------
    >>> nll_fn = make_nll_function(rt_data, choice_data, mu_data, sacc_data, length_data)
    >>> from jax import grad
    >>> grad_nll = grad(nll_fn)
    >>> params = jnp.array([1.0, 1.5, 0.3])  # sigma, a, b
    >>> loss = nll_fn(params)
    >>> grads = grad_nll(params)
    """
    @jit
    def nll_fn(params):
        sigma, a, b = params
        likelihoods = compute_likelihoods_batch(
            rt_data, choice_data, mu_array_data, sacc_array_data, length_data,
            sigma, a, b, x0, order, trunc_num
        )
        return compute_nll_batch_sum(likelihoods)
    
    return nll_fn


# JIT-compiled batch function
# Note: sigma, a, b, x0 are traced, not static, for gradient computation
compute_likelihoods_batch_jit = jit(compute_likelihoods_batch)


# ---------------------------------------------------------------------------
# Chunked (memory-efficient) log-likelihood via lax.scan + jax.checkpoint
# ---------------------------------------------------------------------------

def compute_loglik_chunked(
    rt_data,            # (n_trials,)
    choice_data,        # (n_trials,)
    mu_array_data,      # (n_trials, max_d)
    sacc_array_data,    # (n_trials, max_d)
    length_data,        # (n_trials,)
    sigma, a, b, x0,
    chunk_size,
    order=30,
    trunc_num=50,
    safe_sacc_data=None,
):
    """Memory-efficient log-likelihood via ``lax.scan`` + ``jax.checkpoint``.

    Instead of ``vmap``-ing over all *N* trials at once (which stores
    autodiff intermediates for every trial simultaneously), this function
    processes trials in sequential chunks of size ``chunk_size``.
    ``jax.checkpoint`` ensures that the expensive per-trial intermediates
    (transition matrices, quadrature arrays, series terms) are discarded
    after each chunk and recomputed during the backward pass.

    Parameters
    ----------
    rt_data, choice_data, mu_array_data, sacc_array_data, length_data :
        Same as :func:`compute_likelihoods_batch`.
    sigma, a, b, x0 :
        Model parameters (shared across trials).
    chunk_size : int
        Number of trials per chunk.  Smaller values use less memory but
        increase compute overhead (~2x gradient cost in the limit).
    order, trunc_num :
        Quadrature order and series truncation (see :func:`get_addm_fptd_jax_fast`).
    safe_sacc_data : array of shape (n_trials, max_d), optional
        Pre-computed safely-padded saccade arrays.  If ``None``,
        ``sacc_array_data`` is used directly (caller should ensure safe
        padding).

    Returns
    -------
    total_ll : scalar
        Sum of per-trial log-likelihoods.
    per_trial_ll : array of shape (n_trials,)
        Individual log-likelihood for each trial.
    """
    n_trials = rt_data.shape[0]

    # Pad to exact multiple of chunk_size
    remainder = n_trials % chunk_size
    n_pad = (chunk_size - remainder) % chunk_size
    n_padded = n_trials + n_pad

    def _pad(arr):
        if n_pad == 0:
            return arr
        pad_width = [(0, n_pad)] + [(0, 0)] * (arr.ndim - 1)
        return jnp.pad(arr, pad_width)

    rt_p = _pad(rt_data)
    choice_p = _pad(choice_data)
    mu_p = _pad(mu_array_data)
    sacc_p = _pad(sacc_array_data)
    d_p = _pad(length_data)
    sacc_safe_p = _pad(safe_sacc_data) if safe_sacc_data is not None else sacc_p

    valid = jnp.concatenate([jnp.ones(n_trials), jnp.zeros(n_pad)])

    n_chunks = n_padded // chunk_size

    def _reshape(arr):
        return arr.reshape((n_chunks, chunk_size) + arr.shape[1:])

    chunks = (
        _reshape(rt_p),
        _reshape(choice_p),
        _reshape(mu_p),
        _reshape(sacc_p),
        _reshape(d_p),
        _reshape(sacc_safe_p),
        _reshape(valid),
    )

    def _single_loglik(rt, choice, mu_array, sacc_array, d, safe_sacc):
        fptd = get_addm_fptd_jax_fast(
            rt, d, mu_array, sacc_array, sigma, a, b, x0, choice,
            order=order, trunc_num=trunc_num, safe_sacc=safe_sacc,
        )
        return jnp.log(jnp.maximum(fptd, 1e-30))

    @jax.checkpoint
    def _scan_body(carry, chunk):
        rt_ch, choice_ch, mu_ch, sacc_ch, d_ch, sacc_safe_ch, valid_ch = chunk
        logliks = vmap(_single_loglik)(
            rt_ch, choice_ch, mu_ch, sacc_ch, d_ch, sacc_safe_ch
        )
        masked = logliks * valid_ch
        return carry + jnp.sum(masked), masked

    total_ll, all_logliks = lax.scan(_scan_body, 0.0, chunks)
    per_trial_ll = all_logliks.reshape(-1)[:n_trials]
    return total_ll, per_trial_ll


def make_nll_function_chunked(
    rt_data, choice_data, mu_array_data, sacc_array_data,
    length_data, x0=0.0, chunk_size=2000, order=30, trunc_num=50,
    safe_sacc_data=None,
):
    """Create a chunked NLL function for gradient-based optimization.

    Like :func:`make_nll_function` but uses :func:`compute_loglik_chunked`
    for reduced GPU memory.

    Returns
    -------
    nll_fn : callable
        ``nll_fn(params)`` where ``params = jnp.array([sigma, a, b])``.
    """
    @jit
    def nll_fn(params):
        sigma, a, b = params
        total_ll, _ = compute_loglik_chunked(
            rt_data, choice_data, mu_array_data, sacc_array_data,
            length_data, sigma, a, b, x0,
            chunk_size=chunk_size, order=order, trunc_num=trunc_num,
            safe_sacc_data=safe_sacc_data,
        )
        return -total_ll

    return nll_fn

