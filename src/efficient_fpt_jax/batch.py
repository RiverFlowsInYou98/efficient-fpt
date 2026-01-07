"""
Batch computation of likelihoods using JAX vmap.

Provides efficient batch evaluation of first-passage time densities
over multiple trials.
"""

import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

from .multi_stage import get_addm_fptd_jax


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

