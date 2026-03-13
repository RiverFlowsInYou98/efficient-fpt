"""Tests for the fast Cython aDDM simulator.

Validates the Cython simulator against the Python reference implementation
(``DDModel.simulate_fpt_datum``), checks reproducibility, and benchmarks speed.
"""

import time
import numpy as np
import pytest

from efficient_fpt.addm import aDDModel, simulate_addm, _build_mu_data_padded
from efficient_fpt.addm_simulator_cy import simulate_addm_batch_cy
from efficient_fpt.utils import get_alternating_mu_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_python_batch(
    mu_data, sacc_data, d_data, sigma, a, b, x0, dt, max_t, gaussian_data
):
    """Reference Python simulator using the same pre-generated Gaussians.

    Matches the Cython inner-loop logic exactly so that, given identical
    Gaussians, the outputs should be bitwise identical.
    """
    n_trials = mu_data.shape[0]
    max_steps = gaussian_data.shape[1]
    half_dt = 0.5 * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)

    rt_out = np.full(n_trials, -1.0)
    choice_out = np.zeros(n_trials, dtype=np.int32)

    for trial in range(n_trials):
        d = d_data[trial]
        y = x0
        t_particle = 0.0
        stage = 0

        for step in range(max_steps):
            y += mu_data[trial, stage] * dt + sigma_sqrt_dt * gaussian_data[trial, step]
            t_particle += dt

            while stage + 1 < d and t_particle >= sacc_data[trial, stage + 1]:
                stage += 1

            upper = a - b * t_particle
            if y >= upper:
                rt_out[trial] = t_particle - half_dt
                choice_out[trial] = 1
                break
            elif y <= -upper:
                rt_out[trial] = t_particle - half_dt
                choice_out[trial] = -1
                break

    return rt_out, choice_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchCyMatchesPython:
    """Cython and Python reference must produce identical results for the same
    pre-generated Gaussians."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_identical_outputs(self, seed):
        rng = np.random.default_rng(seed)
        n_trials = 50
        sigma, a, b, x0 = 1.0, 1.5, 0.1, 0.0
        dt, max_t = 1e-3, 10.0
        max_steps = int(max_t / dt) + 1

        mu1 = rng.uniform(-1, 1, n_trials)
        mu2 = rng.uniform(-1, 1, n_trials)
        flag = rng.integers(0, 2, n_trials).astype(np.int32)

        raw_sacc = []
        d_data = np.zeros(n_trials, dtype=np.int32)
        for i in range(n_trials):
            durations = rng.gamma(1.0, 0.3, 100)
            times = np.concatenate([[0.0], np.cumsum(durations)])
            times = times[times < max_t]
            d_data[i] = len(times)
            raw_sacc.append(times)

        max_d = int(d_data.max())
        sacc_data = np.zeros((n_trials, max_d), dtype=np.float64)
        for i in range(n_trials):
            d = d_data[i]
            sacc_data[i, :d] = raw_sacc[i]
        sacc_data = np.ascontiguousarray(sacc_data)
        mu_data = _build_mu_data_padded(mu1, mu2, d_data, flag, max_d)

        gaussian_data = np.ascontiguousarray(
            rng.standard_normal((n_trials, max_steps))
        )

        rt_cy, ch_cy = simulate_addm_batch_cy(
            mu_data, sacc_data, d_data,
            sigma, a, b, x0, dt, max_t,
            gaussian_data,
        )
        rt_py, ch_py = _simulate_python_batch(
            mu_data, sacc_data, d_data,
            sigma, a, b, x0, dt, max_t,
            gaussian_data,
        )

        np.testing.assert_array_equal(ch_cy, ch_py)
        np.testing.assert_allclose(rt_cy, rt_py, atol=1e-12)


class TestSimulateAddm:
    """Integration tests for the high-level ``simulate_addm`` function."""

    def test_basic_run(self):
        result = simulate_addm(
            n_trials=100, eta=0.5, kappa=0.01, sigma=1.0,
            a=1.5, b=0.1, x0=0.0, dt=1e-3, max_t=10.0,
            random_state=42,
        )
        assert result["rt"].shape == (100,)
        assert result["choice"].shape == (100,)
        terminated = result["rt"] > 0
        assert terminated.sum() > 50
        assert set(np.unique(result["choice"][terminated])).issubset({-1, 1})

    def test_reproducibility(self):
        kwargs = dict(
            n_trials=200, eta=0.5, kappa=0.01, sigma=1.0,
            a=1.5, b=0.1, x0=0.0, dt=1e-3, max_t=10.0,
            random_state=12345,
        )
        r1 = simulate_addm(**kwargs)
        r2 = simulate_addm(**kwargs)
        np.testing.assert_array_equal(r1["rt"], r2["rt"])
        np.testing.assert_array_equal(r1["choice"], r2["choice"])

    def test_flat_boundaries(self):
        """b=0 gives constant boundaries."""
        result = simulate_addm(
            n_trials=200, eta=0.5, kappa=0.01, sigma=1.0,
            a=1.5, b=0.0, x0=0.0, dt=1e-3, max_t=10.0,
            random_state=7,
        )
        terminated = result["rt"] > 0
        assert terminated.sum() > 100

    def test_n_threads_error(self):
        with pytest.raises(NotImplementedError):
            simulate_addm(
                n_trials=10, eta=0.5, kappa=0.01, sigma=1.0,
                a=1.5, b=0.1, n_threads=2,
            )

    def test_output_format(self):
        result = simulate_addm(
            n_trials=50, eta=0.5, kappa=0.01, sigma=1.0,
            a=1.5, b=0.1, dt=1e-3, max_t=5.0, random_state=0,
        )
        assert "rt" in result
        assert "choice" in result
        assert "mu_data_padded" in result
        assert "sacc_data_padded" in result
        assert "d_data" in result
        assert "r1" in result
        assert "r2" in result
        assert "flag" in result
        assert "params" in result
        assert result["mu_data_padded"].shape[0] == 50
        assert result["sacc_data_padded"].shape[0] == 50


class TestDistributional:
    """Statistical comparison: the Cython simulator (via simulate_addm) should
    produce RT distributions consistent with the Python aDDModel reference."""

    @pytest.mark.parametrize(
        "eta, kappa, sigma, a, b",
        [
            (0.5, 0.01, 1.0, 1.5, 0.1),
            (0.0, 0.02, 0.8, 1.0, 0.0),
            (1.0, 0.005, 1.2, 2.0, 0.05),
        ],
    )
    def test_rt_distributions_match(self, eta, kappa, sigma, a, b):
        """Simulate with both engines and compare RT CDFs via KS test."""
        from scipy.stats import ks_2samp

        n_trials = 3000
        dt = 1e-3
        max_t = 10.0
        x0 = 0.0
        gamma_shape, gamma_scale = 1.0, 0.3

        cython_result = simulate_addm(
            n_trials=n_trials, eta=eta, kappa=kappa, sigma=sigma,
            a=a, b=b, x0=x0,
            gamma_shape=gamma_shape, gamma_scale=gamma_scale,
            dt=dt, max_t=max_t, random_state=42,
        )

        rng = np.random.default_rng(42)
        py_rts = []
        py_choices = []
        for _ in range(n_trials):
            r1 = rng.integers(1, 6)
            r2 = rng.integers(1, 6)
            flag = rng.integers(0, 2)
            mu1 = kappa * (r1 - eta * r2)
            mu2 = kappa * (eta * r1 - r2)

            fixations = rng.gamma(gamma_shape, gamma_scale, 200)
            sacc = np.concatenate([[0.0], np.cumsum(fixations)])
            sacc = sacc[sacc < max_t]

            model = aDDModel(mu1, mu2, sacc, flag, sigma, a, b, x0)
            rt, choice = model.simulate_fpt_datum(dt=dt)
            if rt < max_t:
                py_rts.append(rt)
                py_choices.append(choice)

        cy_terminated = cython_result["rt"] > 0
        cy_rts = cython_result["rt"][cy_terminated]
        py_rts = np.array(py_rts)

        assert len(cy_rts) > 500, f"Too few Cython terminated trials: {len(cy_rts)}"
        assert len(py_rts) > 500, f"Too few Python terminated trials: {len(py_rts)}"

        stat, p = ks_2samp(cy_rts, py_rts)
        assert p > 0.001, (
            f"KS test failed: stat={stat:.4f}, p={p:.6f}. "
            f"Cython and Python RT distributions differ significantly."
        )


class TestSpeed:
    """Cython simulator should be substantially faster than pure Python."""

    def test_speedup(self):
        n_trials = 200
        dt = 1e-3
        max_t = 5.0
        sigma, a, b, x0 = 1.0, 1.5, 0.1, 0.0
        eta, kappa = 0.5, 0.01

        t0 = time.perf_counter()
        simulate_addm(
            n_trials=n_trials, eta=eta, kappa=kappa, sigma=sigma,
            a=a, b=b, x0=x0, dt=dt, max_t=max_t, random_state=42,
        )
        cy_time = time.perf_counter() - t0

        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        for _ in range(n_trials):
            r1 = rng.integers(1, 6)
            r2 = rng.integers(1, 6)
            flag = rng.integers(0, 2)
            mu1 = kappa * (r1 - eta * r2)
            mu2 = kappa * (eta * r1 - r2)

            fixations = rng.gamma(1.0, 0.3, 200)
            sacc = np.concatenate([[0.0], np.cumsum(fixations)])
            sacc = sacc[sacc < max_t]

            model = aDDModel(mu1, mu2, sacc, flag, sigma, a, b, x0)
            model.simulate_fpt_datum(dt=dt)
        py_time = time.perf_counter() - t0

        speedup = py_time / cy_time
        print(f"\nCython: {cy_time:.3f}s, Python: {py_time:.3f}s, speedup: {speedup:.1f}x")
        assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"
