"""Tests for the chunked (memory-efficient) JAX log-likelihood.

Validates that ``compute_loglik_chunked`` produces identical results
to the standard ``vmap``-based path, both for forward values and gradients.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

from jax import vmap, grad, jit

from efficient_fpt_jax.batch import (
    compute_likelihoods_batch,
    compute_loglik_chunked,
    compute_nll_batch_sum,
)
from efficient_fpt_jax.multi_stage import (
    get_addm_fptd_jax_fast,
    pad_sacc_array_safely,
)
from efficient_fpt.addm import simulate_addm
from efficient_fpt.utils import get_alternating_mu_array


# ---------------------------------------------------------------------------
# Fixture: generate a small synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trial_data():
    """Simulate trials and prepare JAX-compatible arrays."""
    result = simulate_addm(
        n_trials=200, eta=0.5, kappa=0.01, sigma=1.0,
        a=1.5, b=0.1, x0=0.0, dt=1e-3, max_t=10.0,
        random_state=42,
    )

    terminated = result["rt"] > 0
    rt = result["rt"][terminated]
    choice = result["choice"][terminated]
    mu_pad = result["mu_data_padded"][terminated]
    sacc_pad = result["sacc_data_padded"][terminated]
    d = result["d_data"][terminated]

    n_trials = len(rt)
    max_d = mu_pad.shape[1]

    jax_rt = jnp.array(rt, dtype=jnp.float64)
    jax_choice = jnp.array(choice, dtype=jnp.int32)
    jax_mu = jnp.array(mu_pad, dtype=jnp.float64)
    jax_sacc = jnp.array(sacc_pad, dtype=jnp.float64)
    jax_d = jnp.array(d, dtype=jnp.int32)

    jax_sacc_safe = vmap(
        lambda s, dd: pad_sacc_array_safely(s, dd, max_d)
    )(jax_sacc, jax_d)

    return dict(
        rt=jax_rt, choice=jax_choice, mu=jax_mu,
        sacc=jax_sacc, sacc_safe=jax_sacc_safe, d=jax_d,
        n_trials=n_trials, max_d=max_d,
    )


# ---------------------------------------------------------------------------
# Reference: vmap-based total log-likelihood
# ---------------------------------------------------------------------------

def _vmap_total_loglik(rt, choice, mu, sacc_safe, d, sigma, a, b, x0,
                       order=30, trunc_num=30):
    """Reference total log-likelihood using vmap (the existing approach)."""
    def single_ll(rt_i, choice_i, mu_i, sacc_safe_i, d_i):
        fptd = get_addm_fptd_jax_fast(
            rt_i, d_i, mu_i, sacc_safe_i, sigma, a, b, x0, choice_i,
            order=order, trunc_num=trunc_num, safe_sacc=sacc_safe_i,
        )
        return jnp.log(jnp.maximum(fptd, 1e-30))

    logliks = vmap(single_ll)(rt, choice, mu, sacc_safe, d)
    return jnp.sum(logliks), logliks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkedMatchesVmap:
    """Chunked log-likelihood must match the vmap reference exactly."""

    SIGMA, A, B, X0 = 1.0, 1.5, 0.1, 0.0
    TRUNC = 30

    def test_total_loglik_matches(self, trial_data):
        """Total log-likelihood must match to high precision."""
        td = trial_data

        ref_total, _ = _vmap_total_loglik(
            td["rt"], td["choice"], td["mu"], td["sacc_safe"], td["d"],
            self.SIGMA, self.A, self.B, self.X0, trunc_num=self.TRUNC,
        )

        chunked_total, _ = compute_loglik_chunked(
            td["rt"], td["choice"], td["mu"], td["sacc"],
            td["d"], self.SIGMA, self.A, self.B, self.X0,
            chunk_size=50, trunc_num=self.TRUNC,
            safe_sacc_data=td["sacc_safe"],
        )

        np.testing.assert_allclose(
            float(chunked_total), float(ref_total), atol=1e-6,
            err_msg="Total log-likelihood mismatch between vmap and chunked",
        )

    def test_per_trial_matches(self, trial_data):
        """Per-trial log-likelihoods must match to high precision."""
        td = trial_data

        _, ref_per_trial = _vmap_total_loglik(
            td["rt"], td["choice"], td["mu"], td["sacc_safe"], td["d"],
            self.SIGMA, self.A, self.B, self.X0, trunc_num=self.TRUNC,
        )

        _, chunked_per_trial = compute_loglik_chunked(
            td["rt"], td["choice"], td["mu"], td["sacc"],
            td["d"], self.SIGMA, self.A, self.B, self.X0,
            chunk_size=50, trunc_num=self.TRUNC,
            safe_sacc_data=td["sacc_safe"],
        )

        ref_np = np.array(ref_per_trial)
        chunked_np = np.array(chunked_per_trial)
        finite = np.isfinite(ref_np) & np.isfinite(chunked_np)
        np.testing.assert_allclose(
            chunked_np[finite], ref_np[finite], atol=1e-6,
            err_msg="Per-trial log-likelihoods mismatch",
        )
        np.testing.assert_array_equal(
            np.isnan(ref_np), np.isnan(chunked_np),
            err_msg="NaN locations differ between vmap and chunked",
        )

    def test_gradient_matches(self, trial_data):
        """Gradient w.r.t. (sigma, a, b) must match between vmap and chunked."""
        td = trial_data

        def ref_fn(sigma, a, b):
            total, _ = _vmap_total_loglik(
                td["rt"], td["choice"], td["mu"], td["sacc_safe"], td["d"],
                sigma, a, b, self.X0, trunc_num=self.TRUNC,
            )
            return total

        def chunked_fn(sigma, a, b):
            total, _ = compute_loglik_chunked(
                td["rt"], td["choice"], td["mu"], td["sacc"],
                td["d"], sigma, a, b, self.X0,
                chunk_size=50, trunc_num=self.TRUNC,
                safe_sacc_data=td["sacc_safe"],
            )
            return total

        ref_grads = grad(ref_fn, argnums=(0, 1, 2))(
            self.SIGMA, self.A, self.B
        )
        chunked_grads = grad(chunked_fn, argnums=(0, 1, 2))(
            self.SIGMA, self.A, self.B
        )

        for i, name in enumerate(["sigma", "a", "b"]):
            np.testing.assert_allclose(
                float(chunked_grads[i]), float(ref_grads[i]),
                rtol=1e-3, atol=1e-4,
                err_msg=f"Gradient mismatch for {name}",
            )


class TestChunkedEdgeCases:
    """Edge cases for padding and chunk sizes."""

    SIGMA, A, B, X0 = 1.0, 1.5, 0.1, 0.0
    TRUNC = 30

    def test_chunk_size_equals_n(self, trial_data):
        """Single chunk (chunk_size >= N) should match vmap exactly."""
        td = trial_data
        n = td["n_trials"]

        ref_total, _ = _vmap_total_loglik(
            td["rt"], td["choice"], td["mu"], td["sacc_safe"], td["d"],
            self.SIGMA, self.A, self.B, self.X0, trunc_num=self.TRUNC,
        )

        chunked_total, _ = compute_loglik_chunked(
            td["rt"], td["choice"], td["mu"], td["sacc"],
            td["d"], self.SIGMA, self.A, self.B, self.X0,
            chunk_size=n, trunc_num=self.TRUNC,
            safe_sacc_data=td["sacc_safe"],
        )

        np.testing.assert_allclose(
            float(chunked_total), float(ref_total), atol=1e-4,
        )

    def test_non_divisible_chunk_size(self, trial_data):
        """N not divisible by chunk_size should still work (padding)."""
        td = trial_data
        n = td["n_trials"]
        chunk_size = n // 3 + 7  # deliberately non-divisible

        chunked_total, per_trial = compute_loglik_chunked(
            td["rt"], td["choice"], td["mu"], td["sacc"],
            td["d"], self.SIGMA, self.A, self.B, self.X0,
            chunk_size=chunk_size, trunc_num=self.TRUNC,
            safe_sacc_data=td["sacc_safe"],
        )

        assert per_trial.shape[0] == n
        assert jnp.isfinite(chunked_total)

    @pytest.mark.parametrize("chunk_size", [10, 25, 50, 100])
    def test_various_chunk_sizes(self, trial_data, chunk_size):
        """Different chunk sizes should all produce the same total."""
        td = trial_data

        ref_total, _ = _vmap_total_loglik(
            td["rt"], td["choice"], td["mu"], td["sacc_safe"], td["d"],
            self.SIGMA, self.A, self.B, self.X0, trunc_num=self.TRUNC,
        )

        chunked_total, _ = compute_loglik_chunked(
            td["rt"], td["choice"], td["mu"], td["sacc"],
            td["d"], self.SIGMA, self.A, self.B, self.X0,
            chunk_size=chunk_size, trunc_num=self.TRUNC,
            safe_sacc_data=td["sacc_safe"],
        )

        np.testing.assert_allclose(
            float(chunked_total), float(ref_total), atol=1e-4,
            err_msg=f"Mismatch with chunk_size={chunk_size}",
        )
