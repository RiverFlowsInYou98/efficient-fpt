"""Cheap tests for the fast Cython aDDM simulator.

These are intentionally lightweight so they can run safely on a login /
non-compute node. The expensive distributional and timing benchmarks are
excluded from the test suite.
"""

import numpy as np
import pytest

pytest.importorskip("efficient_fpt.cython.simulator")

from efficient_fpt.models import aDDModel, MultiStageModel, SingleStageModel
from efficient_fpt.cython.simulator import (
    _simulate_addm_fpt,
    simulate_homog_ddm_fpt,
)


def _expected_addm_mu_array(mu1, mu2, flag, d):
    first, second = (mu1, mu2) if flag == 0 else (mu2, mu1)
    return np.where(np.arange(d) % 2 == 0, first, second)


def _generated_parts(result):
    return (
        result["decision_data"],
        result["covariates"],
        result["params"],
        result["config"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulateAddm:
    """Integration tests for ``aDDModel.generate_experiment``."""

    def test_basic_run(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)
        result = model.generate_experiment(
            n_trials=24,
            dt=1e-2,
            T=4.0,
            rng=42,
        )
        decision, *_ = _generated_parts(result)
        assert decision["rt_data"].shape == (24,)
        assert decision["choice_data"].shape == (24,)
        terminated = decision["rt_data"] > 0
        assert terminated.sum() >= 8
        assert set(np.unique(decision["choice_data"][terminated])).issubset({-1, 1})

    def test_reproducibility(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)
        kwargs = dict(n_trials=24, dt=1e-2, T=4.0, rng=12345)
        r1 = model.generate_experiment(**kwargs)
        r2 = model.generate_experiment(**kwargs)
        decision1, *_ = _generated_parts(r1)
        decision2, *_ = _generated_parts(r2)
        np.testing.assert_array_equal(decision1["rt_data"], decision2["rt_data"])
        np.testing.assert_array_equal(
            decision1["choice_data"], decision2["choice_data"]
        )

    def test_flat_boundaries(self):
        """b=0 gives constant boundaries."""
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.0, x0=0.0)
        result = model.generate_experiment(
            n_trials=24,
            dt=1e-2,
            T=4.0,
            rng=7,
        )
        decision, *_ = _generated_parts(result)
        terminated = decision["rt_data"] > 0
        assert terminated.sum() >= 8

    def test_n_threads(self):
        """Multi-threaded simulation should produce valid results."""
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)
        result = model.generate_experiment(
            n_trials=24,
            dt=1e-2,
            T=4.0,
            n_threads=2,
            rng=42,
        )
        decision, *_ = _generated_parts(result)
        terminated = decision["rt_data"] > 0
        assert terminated.sum() >= 8

    def test_output_format(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        result = model.generate_experiment(
            n_trials=16,
            dt=1e-2,
            T=3.0,
            rng=0,
        )
        decision, covariates, params, config = _generated_parts(result)
        assert set(result) == {"decision_data", "params", "covariates", "config"}
        assert set(decision) == {"rt_data", "choice_data"}
        assert set(covariates) == {
            "r1_data",
            "r2_data",
            "flag_data",
            "sacc_array_data",
            "d_data",
        }
        assert set(params) == {"eta", "kappa", "sigma", "a", "b", "x0"}
        assert set(config) == {"dt", "T", "gamma_shape", "gamma_scale", "r_range"}
        assert covariates["sacc_array_data"].shape[0] == 16

    def test_returned_trial_metadata_is_self_consistent(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)
        result = model.generate_experiment(
            n_trials=18,
            dt=1e-2,
            T=4.0,
            rng=7,
        )
        decision, covariates, params, config = _generated_parts(result)

        assert set(np.unique(decision["choice_data"])).issubset({-1, 0, 1})
        np.testing.assert_array_equal(
            decision["choice_data"] != 0, decision["rt_data"] > 0
        )

        for trial in range(len(decision["rt_data"])):
            d = int(covariates["d_data"][trial])
            rt = decision["rt_data"][trial]
            choice = int(decision["choice_data"][trial])
            sacc = covariates["sacc_array_data"][trial, :d]
            positive_sacc = sacc[1:][sacc[1:] > 0]

            assert d >= 1
            assert sacc[0] == 0.0
            if positive_sacc.size:
                assert np.all(np.diff(positive_sacc) > 0)

            eta = params["eta"]
            kappa = params["kappa"]
            r1 = covariates["r1_data"][trial]
            r2 = covariates["r2_data"][trial]
            mu1 = kappa * (r1 - eta * r2)
            mu2 = kappa * (eta * r1 - r2)
            expected_mu = _expected_addm_mu_array(
                mu1, mu2, int(covariates["flag_data"][trial]), d
            )
            assert expected_mu.shape == (d,)

            if rt > 0:
                assert choice in {-1, 1}
                assert np.all(positive_sacc < rt)
            else:
                assert rt == -1.0
                assert choice == 0
                assert np.all(positive_sacc < config["T"])


class TestaDDModelClass:
    """Tests for the standalone aDDModel class."""

    def test_init(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)
        assert model.eta == 0.5
        assert model.kappa == 0.01
        assert model.sigma == 1.0
        assert model.a == 1.5
        assert model.b == 0.25
        assert model.x0 == 0.0

    def test_derive_drift_scalar(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        mu1, mu2 = model.derive_drift(5, 3)
        assert mu1 == pytest.approx(0.01 * (5 - 0.5 * 3))
        assert mu2 == pytest.approx(0.01 * (0.5 * 5 - 3))

    def test_derive_drift_array(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        r1 = np.array([3, 5, 2])
        r2 = np.array([4, 1, 6])
        mu1, mu2 = model.derive_drift(r1, r2)
        expected_mu1 = 0.01 * (r1 - 0.5 * r2)
        expected_mu2 = 0.01 * (0.5 * r1 - r2)
        np.testing.assert_allclose(mu1, expected_mu1)
        np.testing.assert_allclose(mu2, expected_mu2)

    def test_simulate_fpt_basic(self):
        """simulate_fpt returns correct shapes and valid outputs."""
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        rng = np.random.default_rng(42)
        n_trials = 20
        max_d = 8

        # Build fake covariates
        r1_data = rng.integers(1, 10, n_trials)
        r2_data = rng.integers(1, 10, n_trials)
        flag_data = rng.integers(0, 2, n_trials).astype(np.int32)
        sacc_array_data = np.zeros((n_trials, max_d))
        d_data = rng.integers(3, max_d + 1, n_trials).astype(np.int32)
        for i in range(n_trials):
            times = np.sort(rng.uniform(0, 3.0, d_data[i] - 1))
            sacc_array_data[i, 0] = 0.0
            sacc_array_data[i, 1 : d_data[i]] = times
            sacc_array_data[i, d_data[i] :] = 0.0

        rt, choice, x_final = model.simulate_fpt(
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            T=4.0,
            dt=1e-2,
            rng=0,
        )

        assert rt.shape == (n_trials,)
        assert choice.shape == (n_trials,)
        assert x_final.shape == (n_trials,)
        # Terminated trials have positive rt and choice in {-1, 1}
        terminated = choice != 0
        assert np.all(rt[terminated] > 0)
        assert set(np.unique(choice[terminated])).issubset({-1, 1})
        # Non-terminated trials have rt == -1 and choice == 0
        assert np.all(rt[~terminated] == -1.0)

    def test_simulate_fpt_reproducible(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        n = 10
        max_d = 5
        rng = np.random.default_rng(99)
        r1_data = rng.integers(1, 10, n)
        r2_data = rng.integers(1, 10, n)
        flag_data = rng.integers(0, 2, n).astype(np.int32)
        d_data = np.full(n, max_d, dtype=np.int32)
        sacc_array_data = np.zeros((n, max_d), dtype=np.float64)
        for i in range(n):
            sacc_array_data[i, 1:] = np.sort(rng.uniform(0, 2.0, max_d - 1))

        rt1, ch1, xf1 = model.simulate_fpt(
            r1_data, r2_data, flag_data, sacc_array_data, d_data, T=3.0, dt=1e-2, rng=42
        )
        rt2, ch2, xf2 = model.simulate_fpt(
            r1_data, r2_data, flag_data, sacc_array_data, d_data, T=3.0, dt=1e-2, rng=42
        )
        np.testing.assert_array_equal(rt1, rt2)
        np.testing.assert_array_equal(ch1, ch2)
        np.testing.assert_array_equal(xf1, xf2)

    def test_simulate_fpt_chunking(self):
        """Results should be identical regardless of chunk_size."""
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        rng = np.random.default_rng(77)
        n = 30
        max_d = 5
        r1_data = rng.integers(1, 10, n)
        r2_data = rng.integers(1, 10, n)
        flag_data = rng.integers(0, 2, n).astype(np.int32)
        d_data = np.full(n, max_d, dtype=np.int32)
        sacc_array_data = np.zeros((n, max_d), dtype=np.float64)
        for i in range(n):
            sacc_array_data[i, 1:] = np.sort(rng.uniform(0, 2.0, max_d - 1))

        rt_big, ch_big, xf_big = model.simulate_fpt(
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            T=3.0,
            dt=1e-2,
            rng=42,
            chunk_size=1000,
        )
        rt_small, ch_small, xf_small = model.simulate_fpt(
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            T=3.0,
            dt=1e-2,
            rng=42,
            chunk_size=7,
        )
        np.testing.assert_array_equal(rt_big, rt_small)
        np.testing.assert_array_equal(ch_big, ch_small)
        np.testing.assert_array_equal(xf_big, xf_small)

    def test_to_multistage_model(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.1)
        sacc = np.array([0.0, 0.3, 0.6, 0.9])
        trial = model.to_multistage_model(0.5, -0.3, sacc)

        assert isinstance(trial, MultiStageModel)
        assert trial.d == 4
        np.testing.assert_allclose(
            trial.mu_array, _expected_addm_mu_array(0.5, -0.3, 0, 4)
        )
        # Symmetric boundaries
        assert trial.a1 == 1.5
        assert trial.a2 == -1.5
        # Can call DDModel methods
        t_grid, X_grids = trial.simulate_trajs(T=2.0, Nt=200, num=10, rng=0)
        assert t_grid.shape == (201,)
        assert X_grids.shape == (10, 201)

    def test_boundary_collapsing_time(self):
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        assert model.boundary_collapsing_time == pytest.approx(6.0)

        model_flat = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.0)
        assert model_flat.boundary_collapsing_time == float("inf")

    def test_ddm_like_interface(self):
        """aDDModel exposes diffusion_coeff, upper_bdy, lower_bdy, initialize_X0."""
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25, x0=0.0)

        assert model.diffusion_coeff(0, 0) == 1.0
        assert model.upper_bdy(0) == pytest.approx(1.5)
        assert model.lower_bdy(0) == pytest.approx(-1.5)
        assert model.upper_bdy(2.0) == pytest.approx(1.0)
        assert model.lower_bdy(2.0) == pytest.approx(-1.0)

        X0 = model.initialize_X0(0.0, 100)
        np.testing.assert_array_equal(X0, np.zeros(100))


class TestInlineRNG:
    """Tests for the inline-RNG Cython simulators."""

    def test_homog_reproducibility(self):
        """Same seeds produce identical results."""
        n = 50
        max_steps = 1001
        dt = 0.001
        drift_vals = np.full(max_steps, 0.5, dtype=np.float64)
        diffusion_vals = np.full(max_steps, 1.0, dtype=np.float64)
        t_grid = np.arange(1, max_steps + 1) * dt
        upper_vals = np.ascontiguousarray(1.5 - 0.25 * t_grid, dtype=np.float64)
        lower_vals = np.ascontiguousarray(-1.5 + 0.25 * t_grid, dtype=np.float64)
        x0_data = np.zeros(n, dtype=np.float64)
        seeds = np.arange(n, dtype=np.uint64)

        T = dt * max_steps
        rt1, ch1, xf1 = simulate_homog_ddm_fpt(
            drift_vals,
            diffusion_vals,
            upper_vals,
            lower_vals,
            x0_data,
            dt,
            max_steps,
            T,
            seeds,
        )
        rt2, ch2, xf2 = simulate_homog_ddm_fpt(
            drift_vals,
            diffusion_vals,
            upper_vals,
            lower_vals,
            x0_data,
            dt,
            max_steps,
            T,
            seeds,
        )
        np.testing.assert_array_equal(rt1, rt2)
        np.testing.assert_array_equal(ch1, ch2)
        np.testing.assert_array_equal(xf1, xf2)

    def test_homog_different_seeds_differ(self):
        """Different seeds produce different results."""
        n = 100
        max_steps = 1001
        dt = 0.001
        drift_vals = np.full(max_steps, 0.5, dtype=np.float64)
        diffusion_vals = np.full(max_steps, 1.0, dtype=np.float64)
        t_grid = np.arange(1, max_steps + 1) * dt
        upper_vals = np.ascontiguousarray(1.5 - 0.25 * t_grid, dtype=np.float64)
        lower_vals = np.ascontiguousarray(-1.5 + 0.25 * t_grid, dtype=np.float64)
        x0_data = np.zeros(n, dtype=np.float64)

        seeds_a = np.arange(n, dtype=np.uint64)
        seeds_b = np.arange(1000, 1000 + n, dtype=np.uint64)
        T = dt * max_steps
        rt_a, ch_a, _ = simulate_homog_ddm_fpt(
            drift_vals,
            diffusion_vals,
            upper_vals,
            lower_vals,
            x0_data,
            dt,
            max_steps,
            T,
            seeds_a,
        )
        rt_b, ch_b, _ = simulate_homog_ddm_fpt(
            drift_vals,
            diffusion_vals,
            upper_vals,
            lower_vals,
            x0_data,
            dt,
            max_steps,
            T,
            seeds_b,
        )
        assert not np.array_equal(rt_a, rt_b)

    def test_ddmodel_uses_inline_rng(self):
        """SingleStageModel.simulate_fpt now uses inline RNG and is reproducible."""
        model = SingleStageModel(mu=0.5, sigma=1.0, a=1.5, b=0.25, x0=0.0)
        rt1, ch1, xf1 = model.simulate_fpt(100, T=1.0, dt=1e-3, rng=42)
        rt2, ch2, xf2 = model.simulate_fpt(100, T=1.0, dt=1e-3, rng=42)
        np.testing.assert_array_equal(rt1, rt2)
        np.testing.assert_array_equal(ch1, ch2)
        np.testing.assert_array_equal(xf1, xf2)

    def test_ddmodel_reasonable_distribution(self):
        """SingleStageModel simulation produces sensible RT distribution."""
        model = SingleStageModel(mu=0.5, sigma=1.0, a=1.5, b=0.0, x0=0.0)
        rt, choice, _ = model.simulate_fpt(2000, T=10.0, dt=1e-3, rng=0)
        terminated = rt > 0
        # Most trials should terminate
        assert terminated.sum() > 1500
        # With positive drift, more upper boundary crossings
        assert (choice[terminated] == 1).sum() > (choice[terminated] == -1).sum()
        # Mean RT should be reasonable (not near 0 or near T)
        mean_rt = rt[terminated].mean()
        assert 0.1 < mean_rt < 5.0

    def test_addm_inline_rng_reproducibility(self):
        """aDDM inline RNG simulator is reproducible with same seeds."""
        rng = np.random.default_rng(42)
        n = 20
        max_d = 5
        mu_array_data = np.ascontiguousarray(
            rng.uniform(-0.5, 0.5, (n, max_d)), dtype=np.float64
        )
        sacc_array_data = np.zeros((n, max_d), dtype=np.float64)
        d_data = np.full(n, max_d, dtype=np.int32)
        for i in range(n):
            sacc_array_data[i, 1:] = np.sort(rng.uniform(0, 2.0, max_d - 1))
        sacc_array_data = np.ascontiguousarray(sacc_array_data, dtype=np.float64)

        seeds = np.arange(n, dtype=np.uint64)
        x0_data = np.zeros(n, dtype=np.float64)

        rt1, ch1, xf1 = _simulate_addm_fpt(
            mu_array_data,
            sacc_array_data,
            d_data,
            1.0,
            1.5,
            0.25,
            x0_data,
            1e-2,
            3.0,
            seeds,
        )
        rt2, ch2, xf2 = _simulate_addm_fpt(
            mu_array_data,
            sacc_array_data,
            d_data,
            1.0,
            1.5,
            0.25,
            x0_data,
            1e-2,
            3.0,
            seeds,
        )
        np.testing.assert_array_equal(rt1, rt2)
        np.testing.assert_array_equal(ch1, ch2)
        np.testing.assert_array_equal(xf1, xf2)

    def test_memory_efficiency(self):
        """Inline RNG should not allocate a massive Gaussian matrix."""
        import tracemalloc

        model = SingleStageModel(mu=0.5, sigma=1.0, a=1.5, b=0.25, x0=0.0)

        tracemalloc.start()
        rt, choice, _ = model.simulate_fpt(10000, T=5.0, dt=1e-3, rng=0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Old approach: 10000 * 5001 * 8 bytes = ~400 MB
        # New approach: should be well under 100 MB
        assert (
            peak < 100 * 1024 * 1024
        ), f"Peak memory {peak / 1024**2:.1f} MB exceeds 100 MB"

    def test_chunk_size_independence(self):
        """aDDModel results identical regardless of chunk_size with inline RNG."""
        model = aDDModel(eta=0.5, kappa=0.01, sigma=1.0, a=1.5, b=0.25)
        rng = np.random.default_rng(77)
        n = 30
        max_d = 5
        r1_data = rng.integers(1, 10, n)
        r2_data = rng.integers(1, 10, n)
        flag_data = rng.integers(0, 2, n).astype(np.int32)
        d_data = np.full(n, max_d, dtype=np.int32)
        sacc_array_data = np.zeros((n, max_d), dtype=np.float64)
        for i in range(n):
            sacc_array_data[i, 1:] = np.sort(rng.uniform(0, 2.0, max_d - 1))

        rt_big, ch_big, xf_big = model.simulate_fpt(
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            T=3.0,
            dt=1e-2,
            rng=42,
            chunk_size=1000,
        )
        rt_small, ch_small, xf_small = model.simulate_fpt(
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            T=3.0,
            dt=1e-2,
            rng=42,
            chunk_size=7,
        )
        np.testing.assert_array_equal(rt_big, rt_small)
        np.testing.assert_array_equal(ch_big, ch_small)
        np.testing.assert_array_equal(xf_big, xf_small)
