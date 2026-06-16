"""
Equivalence tests for JAX single-stage implementation.

Compares the JAX implementation against the original NumPy implementation
to ensure numerical equivalence.
"""

import pytest
import numpy as np

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax import vmap

from efficient_fpt.single_stage import fptd_basic, q_basic, fptd_single, q_single
from efficient_fpt.jax.single_stage import (
    fptd_basic as fptd_basic_jax,
    q_basic as q_basic_jax,
    fptd_single as fptd_single_jax,
    q_single as q_single_jax,
)


class TestFPTDBasicEquivalence:
    """Test equivalence of fptd_basic implementations."""
    
    # Tolerances account for float32 (JAX default) vs float64 (NumPy) differences
    RTOL = 1e-5
    ATOL = 1e-7
    TRUNC_NUM = 50  # Fixed for both implementations
    
    @pytest.fixture
    def standard_params(self):
        return {
            'mu': 0.5, 'a1': 1.0, 'b1': -0.3, 'a2': -1.0, 'b2': 0.3
        }
    
    @pytest.mark.parametrize("t", [0.1, 0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("bdy", [1, -1])
    def test_fptd_basic_equivalence(self, standard_params, t, bdy):
        """Test fptd_basic matches across implementations."""
        np_result = fptd_basic(
            t, **standard_params, bdy=bdy, 
            trunc_num=self.TRUNC_NUM, adaptive_stopping=False
        )
        jax_result = fptd_basic_jax(
            t, **standard_params, bdy=bdy, 
            trunc_num=self.TRUNC_NUM
        )
        np.testing.assert_allclose(
            np_result, float(jax_result), 
            rtol=self.RTOL, atol=self.ATOL
        )
    
    @pytest.mark.parametrize("mu", [-1.0, 0.0, 0.5, 2.0])
    @pytest.mark.parametrize("a1", [0.5, 1.0, 2.0])
    def test_fptd_basic_varied_params(self, mu, a1):
        """Test with varied parameter combinations."""
        params = {'mu': mu, 'a1': a1, 'b1': -0.3, 'a2': -a1, 'b2': 0.3}
        t = 1.0
        bdy = 1
        
        np_result = fptd_basic(t, **params, bdy=bdy, 
                               trunc_num=self.TRUNC_NUM, adaptive_stopping=False)
        jax_result = fptd_basic_jax(t, **params, bdy=bdy, 
                                     trunc_num=self.TRUNC_NUM)
        
        np.testing.assert_allclose(np_result, float(jax_result),
                                   rtol=self.RTOL, atol=self.ATOL)
    
    def test_fptd_basic_vectorized_t(self, standard_params):
        """Test vectorized evaluation over time array."""
        t_array = np.array([0.1, 0.5, 1.0, 2.0])
        bdy = 1
        
        # NumPy version (loop)
        np_results = np.array([
            fptd_basic(t, **standard_params, bdy=bdy, 
                      trunc_num=self.TRUNC_NUM, adaptive_stopping=False)
            for t in t_array
        ])
        
        # JAX version (vmap)
        jax_fn = lambda t: fptd_basic_jax(
            t, standard_params['mu'], standard_params['a1'], 
            standard_params['b1'], standard_params['a2'], 
            standard_params['b2'], bdy, trunc_num=self.TRUNC_NUM
        )
        jax_results = vmap(jax_fn)(jnp.array(t_array))
        
        np.testing.assert_allclose(np_results, np.array(jax_results), 
                                   rtol=self.RTOL, atol=self.ATOL)


class TestQBasicEquivalence:
    """Test equivalence of q_basic implementations."""
    
    # Tolerances account for float32 (JAX default) vs float64 (NumPy) differences
    RTOL = 1e-5
    ATOL = 1e-7
    TRUNC_NUM = 50
    
    @pytest.fixture
    def standard_params(self):
        return {
            'mu': 0.5, 'a1': 1.0, 'b1': -0.3, 'a2': -1.0, 'b2': 0.3, 'T': 1.0
        }
    
    @pytest.mark.parametrize("x", [-0.5, -0.2, 0.0, 0.2, 0.5])
    def test_q_basic_equivalence(self, standard_params, x):
        """Test q_basic matches across implementations."""
        np_result = q_basic(
            x, **standard_params,
            trunc_num=self.TRUNC_NUM, adaptive_stopping=False
        )
        jax_result = q_basic_jax(
            x, **standard_params,
            trunc_num=self.TRUNC_NUM
        )
        np.testing.assert_allclose(
            np_result, float(jax_result), 
            rtol=self.RTOL, atol=self.ATOL
        )
    
    def test_q_basic_vectorized_x(self, standard_params):
        """Test vectorized evaluation over position array."""
        x_array = np.linspace(-0.5, 0.5, 20)
        
        # NumPy version (loop)
        np_results = np.array([
            q_basic(x, **standard_params, 
                   trunc_num=self.TRUNC_NUM, adaptive_stopping=False)
            for x in x_array
        ])
        
        # JAX version (direct broadcasting)
        jax_results = q_basic_jax(
            jnp.array(x_array), 
            standard_params['mu'], standard_params['a1'], 
            standard_params['b1'], standard_params['a2'], 
            standard_params['b2'], standard_params['T'], trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(np_results, np.array(jax_results), 
                                   rtol=self.RTOL, atol=self.ATOL)


class TestBroadcastingBehavior:
    """Test that broadcasting works correctly for batch computation."""
    
    # Tolerances account for float32 (JAX default) vs float64 (NumPy) differences
    RTOL = 1e-5
    ATOL = 1e-7
    TRUNC_NUM = 50
    
    def test_q_single_transition_matrix(self):
        """Test broadcasting produces correct transition matrix."""
        order = 10
        mu, sigma = 0.5, 1.0
        a1, b1 = 1.0, -0.3
        a2, b2 = -1.0, 0.3
        T = 0.5
        
        xs = jnp.linspace(-0.5, 0.5, order)      # destinations
        xs_prev = jnp.linspace(-0.6, 0.6, order) # sources
        
        # Broadcasted call
        P_broadcast = q_single_jax(
            xs[:, None],      # (order, 1)
            mu, sigma, a1, b1, a2, b2, T,
            xs_prev[None, :], # (1, order)
            trunc_num=self.TRUNC_NUM
        )  # Should be (order, order)
        
        assert P_broadcast.shape == (order, order)
        
        # Loop-based reference
        P_loop = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                P_loop[i, j] = float(q_single_jax(
                    float(xs[i]), mu, sigma, a1, b1, a2, b2, T, 
                    float(xs_prev[j]), trunc_num=self.TRUNC_NUM
                ))
        
        np.testing.assert_allclose(np.array(P_broadcast), P_loop, 
                                   rtol=self.RTOL, atol=self.ATOL)
    
    def test_fptd_single_batch_x0(self):
        """Test FPTD broadcasting over starting positions."""
        t = 1.0
        mu, sigma = 0.5, 1.0
        a1, b1 = 1.0, -0.3
        a2, b2 = -1.0, 0.3
        bdy = 1
        
        x0_array = jnp.linspace(-0.5, 0.5, 20)
        
        # Broadcasted
        fptds_broadcast = fptd_single_jax(
            t, mu, sigma, a1, b1, a2, b2, x0_array, bdy, trunc_num=self.TRUNC_NUM
        )
        
        # Loop reference
        fptds_loop = jnp.array([
            fptd_single_jax(
                t, mu, sigma, a1, b1, a2, b2, float(x0), bdy, trunc_num=self.TRUNC_NUM
            )
            for x0 in x0_array
        ])
        
        np.testing.assert_allclose(np.array(fptds_broadcast), np.array(fptds_loop), 
                                   rtol=self.RTOL, atol=self.ATOL)


class TestSingleStageWrapperEquivalence:
    """Test fptd_single and q_single with sigma scaling."""
    
    # Tolerances account for float32 (JAX default) vs float64 (NumPy) differences
    RTOL = 1e-5
    ATOL = 1e-7
    TRUNC_NUM = 50
    
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("x0", [-0.3, 0.0, 0.3])
    def test_fptd_single_equivalence(self, sigma, x0):
        """Test fptd_single with various sigma and x0."""
        t = 1.0
        mu = 0.5
        a1, b1 = 1.5, -0.3
        a2, b2 = -1.5, 0.3
        bdy = 1
        
        np_result = fptd_single(
            t, mu, sigma, a1, b1, a2, b2, x0, bdy,
            trunc_num=self.TRUNC_NUM, adaptive_stopping=False
        )
        jax_result = fptd_single_jax(
            t, mu, sigma, a1, b1, a2, b2, x0, bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(np_result, float(jax_result),
                                   rtol=self.RTOL, atol=self.ATOL)
    
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("x0", [-0.3, 0.0, 0.3])
    def test_q_single_equivalence(self, sigma, x0):
        """Test q_single with various sigma and x0."""
        x = 0.2
        T = 1.0
        mu = 0.5
        a1, b1 = 1.5, -0.3
        a2, b2 = -1.5, 0.3
        
        np_result = q_single(
            x, mu, sigma, a1, b1, a2, b2, T, x0,
            trunc_num=self.TRUNC_NUM, adaptive_stopping=False
        )
        jax_result = q_single_jax(
            x, mu, sigma, a1, b1, a2, b2, T, x0, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(np_result, float(jax_result),
                                   rtol=self.RTOL, atol=self.ATOL)


# ---------------------------------------------------------------------------
# JAX log functions (migrated from test_untested_coverage.py)
# ---------------------------------------------------------------------------

from efficient_fpt.jax.single_stage import (
    log_fptd_basic,
    log_q_basic,
    log_fptd_single,
    log_q_single,
)


class TestJAXLogFunctions:
    @pytest.fixture(autouse=True)
    def _import_jax(self):
        self.fptd_basic = fptd_basic_jax
        self.q_basic = q_basic_jax
        self.fptd_single = fptd_single_jax
        self.q_single = q_single_jax
        self.log_fptd_basic = log_fptd_basic
        self.log_q_basic = log_q_basic
        self.log_fptd_single = log_fptd_single
        self.log_q_single = log_q_single

    def test_log_fptd_basic_matches_log_of_fptd(self):
        t = 0.5
        mu, a1, b1, a2, b2 = 1.0, 1.0, -0.5, -1.0, 0.5
        for bdy in [1, -1]:
            density = self.fptd_basic(t, mu, a1, b1, a2, b2, bdy)
            log_density = self.log_fptd_basic(t, mu, a1, b1, a2, b2, bdy)
            np.testing.assert_allclose(
                float(log_density), float(jnp.log(density)), rtol=1e-10
            )

    def test_log_q_basic_matches_log_of_q(self):
        x = 0.3
        mu, a1, b1, a2, b2, T = 1.0, 1.0, -0.5, -1.0, 0.5, 0.5
        density = self.q_basic(x, mu, a1, b1, a2, b2, T)
        log_density = self.log_q_basic(x, mu, a1, b1, a2, b2, T)
        np.testing.assert_allclose(
            float(log_density), float(jnp.log(density)), rtol=1e-10
        )

    def test_log_fptd_single_matches_log_of_fptd(self):
        t = 0.5
        mu, sigma, a1, b1, a2, b2, x0 = 1.0, 1.5, 1.0, -0.5, -1.0, 0.5, 0.1
        for bdy in [1, -1]:
            density = self.fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy)
            log_density = self.log_fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy)
            np.testing.assert_allclose(
                float(log_density), float(jnp.log(density)), rtol=1e-10
            )

    def test_log_q_single_matches_log_of_q(self):
        x = 0.3
        mu, sigma, a1, b1, a2, b2, T, x0 = 1.0, 1.5, 1.0, -0.5, -1.0, 0.5, 0.5, 0.1
        density = self.q_single(x, mu, sigma, a1, b1, a2, b2, T, x0)
        log_density = self.log_q_single(x, mu, sigma, a1, b1, a2, b2, T, x0)
        np.testing.assert_allclose(
            float(log_density), float(jnp.log(density)), rtol=1e-10
        )

    def test_log_fptd_basic_array(self):
        t = jnp.array([0.3, 0.5, 0.8])
        mu, a1, b1, a2, b2 = 1.0, 1.0, -0.5, -1.0, 0.5
        log_density = self.log_fptd_basic(t, mu, a1, b1, a2, b2, 1)
        assert log_density.shape == (3,)
        assert jnp.all(jnp.isfinite(log_density))

    def test_log_returns_neg_inf_for_zero_density(self):
        # At t=0, density should be 0, log should be -inf
        log_d = self.log_fptd_basic(0.0, 1.0, 1.0, -0.5, -1.0, 0.5, 1)
        assert float(log_d) == float("-inf")


# ---------------------------------------------------------------------------
# JAX lgwt_lookup_table (migrated from test_untested_coverage.py)
# ---------------------------------------------------------------------------


class TestLgwtLookupTable:
    def test_correct_length(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        x, w = lgwt_lookup_table(10, 0.0, 1.0)
        assert x.shape == (10,)
        assert w.shape == (10,)

    def test_nodes_in_interval(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        x, w = lgwt_lookup_table(20, -2.0, 3.0)
        assert float(jnp.min(x)) >= -2.0
        assert float(jnp.max(x)) <= 3.0

    def test_weights_sum_to_interval_length(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        x, w = lgwt_lookup_table(30, 2.0, 5.0)
        np.testing.assert_allclose(float(jnp.sum(w)), 3.0, rtol=1e-12)

    def test_integrates_polynomial_exactly(self):
        from efficient_fpt.jax.utils import lgwt_lookup_table

        # Order-n Gauss-Legendre integrates polynomials up to degree 2n-1 exactly
        x, w = lgwt_lookup_table(5, 0.0, 1.0)
        # Integral of x^4 from 0 to 1 = 1/5 (degree 4 < 2*5-1=9)
        result = float(jnp.sum(w * x**4))
        np.testing.assert_allclose(result, 0.2, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
