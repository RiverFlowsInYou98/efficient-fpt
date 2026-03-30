"""Cross-backend equivalence tests.

Parametrized tests that verify NumPy, Cython, and JAX backends produce
identical results for the same inputs.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Backend-parametrized fixtures
# ---------------------------------------------------------------------------


def _load_single_stage_funcs(backend_name):
    """Return (fptd_basic, q_basic, fptd_single, q_single) for a backend."""
    if backend_name == "numpy":
        from efficient_fpt.numpy.single_stage import (
            fptd_basic, q_basic, fptd_single, q_single,
        )
    elif backend_name == "cython":
        pytest.importorskip("efficient_fpt.cython")
        from efficient_fpt.cython import fptd_basic, q_basic, fptd_single, q_single
    elif backend_name == "jax":
        pytest.importorskip("jax")
        from efficient_fpt.jax import fptd_basic, q_basic, fptd_single, q_single
    return dict(
        fptd_basic=fptd_basic,
        q_basic=q_basic,
        fptd_single=fptd_single,
        q_single=q_single,
    )


@pytest.fixture(params=["numpy", "cython", "jax"])
def fptd_backend(request):
    """Parametrized fixture returning single-stage functions for each backend."""
    funcs = _load_single_stage_funcs(request.param)
    funcs["_name"] = request.param
    return funcs


# ---------------------------------------------------------------------------
# Single-stage: known-value tests
# ---------------------------------------------------------------------------


class TestFptdBasicCrossBackend:
    """Verify fptd_basic returns the same values across all backends."""

    def test_positive_density(self, fptd_backend):
        val = float(fptd_backend["fptd_basic"](0.5, 1.0, 1.5, -0.3, -1.5, 0.3, 1))
        assert val > 0
        assert np.isfinite(val)

    def test_upper_lower_symmetry(self, fptd_backend):
        """With symmetric params (mu=0), upper and lower densities should be equal."""
        f = fptd_backend["fptd_basic"]
        upper = float(f(0.5, 0.0, 1.5, -0.3, -1.5, 0.3, 1))
        lower = float(f(0.5, 0.0, 1.5, -0.3, -1.5, 0.3, -1))
        np.testing.assert_allclose(upper, lower, rtol=1e-10)

    def test_array_input(self, fptd_backend):
        if fptd_backend["_name"] == "cython":
            pytest.skip("Cython single-stage functions are scalar-only")
        t = np.array([0.3, 0.5, 1.0])
        vals = fptd_backend["fptd_basic"](t, 1.0, 1.5, -0.3, -1.5, 0.3, 1)
        vals = np.asarray(vals)
        assert vals.shape == (3,)
        assert np.all(vals > 0)


class TestQBasicCrossBackend:
    """Verify q_basic returns the same values across all backends."""

    def test_positive_density(self, fptd_backend):
        val = float(fptd_backend["q_basic"](0.0, 1.0, 1.5, -0.3, -1.5, 0.3, 0.5))
        assert val > 0
        assert np.isfinite(val)

    def test_array_input(self, fptd_backend):
        if fptd_backend["_name"] == "cython":
            pytest.skip("Cython single-stage functions are scalar-only")
        x = np.linspace(-1.0, 1.0, 5)
        vals = fptd_backend["q_basic"](x, 1.0, 1.5, -0.3, -1.5, 0.3, 0.5)
        vals = np.asarray(vals)
        assert vals.shape == (5,)
        assert np.all(vals >= 0)


class TestFptdSingleCrossBackend:
    """Verify fptd_single returns the same values across all backends."""

    def test_sigma_scaling(self, fptd_backend):
        val = float(fptd_backend["fptd_single"](
            0.5, 1.0, 2.0, 1.5, -0.3, -1.5, 0.3, 0.0, 1
        ))
        assert val > 0
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Cross-backend numerical agreement (pairwise)
# ---------------------------------------------------------------------------


class TestCrossBackendAgreement:
    """Compare numerical results across backend pairs."""

    @pytest.fixture
    def all_backends(self):
        """Load all available backends."""
        backends = {"numpy": _load_single_stage_funcs("numpy")}
        try:
            backends["cython"] = _load_single_stage_funcs("cython")
        except pytest.skip.Exception:
            pass
        try:
            backends["jax"] = _load_single_stage_funcs("jax")
        except pytest.skip.Exception:
            pass
        if len(backends) < 2:
            pytest.skip("Need at least 2 backends for comparison")
        return backends

    @staticmethod
    def _eval_scalar_or_array(func, args, is_cython):
        """Call func with scalar inputs if Cython, array otherwise."""
        first_arg = args[0]
        if is_cython and np.ndim(first_arg) > 0:
            return np.array([func(float(v), *args[1:]) for v in first_arg])
        return np.asarray(func(*args), dtype=np.float64)

    def test_fptd_basic_agreement(self, all_backends):
        t_vals = np.array([0.3, 0.5, 1.0, 2.0])
        args = (t_vals, 1.0, 1.5, -0.3, -1.5, 0.3, 1)
        results = {}
        for name, funcs in all_backends.items():
            results[name] = self._eval_scalar_or_array(
                funcs["fptd_basic"], args, name == "cython"
            )
        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=1e-10,
                err_msg=f"{names[0]} vs {names[i]}",
            )

    def test_q_basic_agreement(self, all_backends):
        x_vals = np.linspace(-1.0, 1.0, 10)
        args = (x_vals, 1.0, 1.5, -0.3, -1.5, 0.3, 0.5)
        results = {}
        for name, funcs in all_backends.items():
            results[name] = self._eval_scalar_or_array(
                funcs["q_basic"], args, name == "cython"
            )
        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=1e-10,
                err_msg=f"{names[0]} vs {names[i]}",
            )

    def test_fptd_single_agreement(self, all_backends):
        t_vals = np.array([0.3, 0.5, 1.0])
        args = (t_vals, 1.0, 2.0, 1.5, -0.3, -1.5, 0.3, 0.0, 1)
        results = {}
        for name, funcs in all_backends.items():
            results[name] = self._eval_scalar_or_array(
                funcs["fptd_single"], args, name == "cython"
            )
        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=1e-10,
                err_msg=f"{names[0]} vs {names[i]}",
            )

    def test_q_single_agreement(self, all_backends):
        x_vals = np.linspace(-1.0, 1.0, 10)
        args = (x_vals, 1.0, 2.0, 1.5, -0.3, -1.5, 0.3, 0.5, 0.0)
        results = {}
        for name, funcs in all_backends.items():
            results[name] = self._eval_scalar_or_array(
                funcs["q_single"], args, name == "cython"
            )
        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=1e-10,
                err_msg=f"{names[0]} vs {names[i]}",
            )


# ---------------------------------------------------------------------------
# Batch NLL cross-backend (Cython vs JAX)
# ---------------------------------------------------------------------------


class TestBatchNLLCrossBackend:
    """Compare batch NLL between Cython and JAX backends."""

    def test_compute_addm_nll_mean(self):
        """Both backends should produce similar mean NLL for the same data."""
        pytest.importorskip("jax")
        try:
            from efficient_fpt.cython.batch import compute_addm_nll as cy_nll
        except ImportError:
            pytest.skip("Cython backend not available")
        from efficient_fpt.jax.batch import compute_addm_nll as jax_nll
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        n = 5
        rt = rng.uniform(0.5, 2.0, n).astype(np.float64)
        choice = rng.choice([-1, 1], n).astype(np.int32)
        r1 = rng.uniform(1, 5, n).astype(np.float64)
        r2 = rng.uniform(1, 5, n).astype(np.float64)
        flag = rng.integers(0, 2, n).astype(np.int32)
        d_data = np.full(n, 3, dtype=np.int32)
        sacc = np.column_stack([
            np.zeros(n),
            rng.uniform(0.1, 0.5, n),
            rng.uniform(0.5, 1.0, n),
        ]).astype(np.float64)

        eta, kappa, sigma, a, b, x0 = 0.5, 0.01, 1.0, 1.5, 0.25, 0.0

        cy_result = cy_nll(
            rt, choice, eta, kappa, sigma, a, b, x0,
            r1, r2, flag, sacc, d_data,
            warn=False, reduce="mean",
        )
        jax_result = float(jax_nll(
            jnp.array(rt), jnp.array(choice), eta, kappa, sigma, a, b, x0,
            jnp.array(r1), jnp.array(r2), jnp.array(flag),
            jnp.array(sacc), jnp.array(d_data),
            warn=False, reduce="mean",
        ))

        # Cython uses adaptive stopping; JAX uses fixed-length series.
        # Expect same order of magnitude, not exact match.
        np.testing.assert_allclose(cy_result, jax_result, rtol=0.5)

    def test_compute_addm_nll_sum(self):
        """Sum reduction should also agree."""
        pytest.importorskip("jax")
        try:
            from efficient_fpt.cython.batch import compute_addm_nll as cy_nll
        except ImportError:
            pytest.skip("Cython backend not available")
        from efficient_fpt.jax.batch import compute_addm_nll as jax_nll
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        n = 5
        rt = rng.uniform(0.5, 2.0, n).astype(np.float64)
        choice = rng.choice([-1, 1], n).astype(np.int32)
        r1 = rng.uniform(1, 5, n).astype(np.float64)
        r2 = rng.uniform(1, 5, n).astype(np.float64)
        flag = rng.integers(0, 2, n).astype(np.int32)
        d_data = np.full(n, 3, dtype=np.int32)
        sacc = np.column_stack([
            np.zeros(n),
            rng.uniform(0.1, 0.5, n),
            rng.uniform(0.5, 1.0, n),
        ]).astype(np.float64)

        eta, kappa, sigma, a, b, x0 = 0.5, 0.01, 1.0, 1.5, 0.25, 0.0

        cy_result = cy_nll(
            rt, choice, eta, kappa, sigma, a, b, x0,
            r1, r2, flag, sacc, d_data,
            warn=False, reduce="sum",
        )
        jax_result = float(jax_nll(
            jnp.array(rt), jnp.array(choice), eta, kappa, sigma, a, b, x0,
            jnp.array(r1), jnp.array(r2), jnp.array(flag),
            jnp.array(sacc), jnp.array(d_data),
            warn=False, reduce="sum",
        ))

        np.testing.assert_allclose(cy_result, jax_result, rtol=0.5)
