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
from efficient_fpt_jax.single_stage import (
    fptd_basic_jax, q_basic_jax, fptd_single_jax, q_single_jax
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
            trunc_num=self.TRUNC_NUM, fixed_terms=True
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
                               trunc_num=self.TRUNC_NUM, fixed_terms=True)
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
                      trunc_num=self.TRUNC_NUM, fixed_terms=True)
            for t in t_array
        ])
        
        # JAX version (vmap)
        jax_fn = lambda t: fptd_basic_jax(
            t, standard_params['mu'], standard_params['a1'], 
            standard_params['b1'], standard_params['a2'], 
            standard_params['b2'], bdy, self.TRUNC_NUM
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
            trunc_num=self.TRUNC_NUM, fixed_terms=True
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
                   trunc_num=self.TRUNC_NUM, fixed_terms=True)
            for x in x_array
        ])
        
        # JAX version (direct broadcasting)
        jax_results = q_basic_jax(
            jnp.array(x_array), 
            standard_params['mu'], standard_params['a1'], 
            standard_params['b1'], standard_params['a2'], 
            standard_params['b2'], standard_params['T'],
            self.TRUNC_NUM
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
            self.TRUNC_NUM
        )  # Should be (order, order)
        
        assert P_broadcast.shape == (order, order)
        
        # Loop-based reference
        P_loop = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                P_loop[i, j] = float(q_single_jax(
                    float(xs[i]), mu, sigma, a1, b1, a2, b2, T, 
                    float(xs_prev[j]), self.TRUNC_NUM
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
            t, mu, sigma, a1, b1, a2, b2, x0_array, bdy, self.TRUNC_NUM
        )
        
        # Loop reference
        fptds_loop = jnp.array([
            fptd_single_jax(t, mu, sigma, a1, b1, a2, b2, float(x0), bdy, self.TRUNC_NUM)
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
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        jax_result = fptd_single_jax(
            t, mu, sigma, a1, b1, a2, b2, x0, bdy, self.TRUNC_NUM
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
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        jax_result = q_single_jax(
            x, mu, sigma, a1, b1, a2, b2, T, x0, self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(np_result, float(jax_result),
                                   rtol=self.RTOL, atol=self.ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

