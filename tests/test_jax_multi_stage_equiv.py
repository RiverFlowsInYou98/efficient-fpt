"""
Equivalence tests for JAX multi-stage implementation.

Compares the JAX implementation against a Python reference implementation
and (optionally) the Cython implementation.
"""

import pytest
import numpy as np

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax import vmap

from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.multi_stage import lgwtLookupTable
from efficient_fpt_jax.multi_stage import get_addm_fptd_jax


def get_addm_fptd_python(t, d, mu_array, sacc_array, sigma, a, b, x0, bdy, 
                          trunc_num=50, fixed_terms=True):
    """
    Pure Python reference implementation of get_addm_fptd.
    
    This is a direct translation of the Cython algorithm into pure Python,
    using the fixed_terms flag for fair comparison with JAX.
    """
    order = 30
    threshold = 1e-30  # Effectively disabled
    
    if d == 1:
        return fptd_single(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, 
                          trunc_num, threshold, fixed_terms)
    
    # Multi-stage case
    x_ref, w_ref = lgwtLookupTable(order, -1.0, 1.0)
    
    # Initialize from stage 0 to stage 1
    a_1 = a - b * sacc_array[1]
    xs = x_ref * a_1
    ws = w_ref * a_1
    
    # Compute distribution at end of stage 0
    pv = np.array([
        q_single(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0,
                trunc_num, threshold, fixed_terms)
        for i in range(order)
    ])
    
    xs_prev = xs.copy()
    ws_pv_prev = ws * pv
    
    # Propagate through intermediate stages
    for n in range(2, d):
        a_curr = a - b * sacc_array[n]
        xs = x_ref * a_curr
        ws = w_ref * a_curr
        
        a_prev = a - b * sacc_array[n-1]
        T_curr = sacc_array[n] - sacc_array[n-1]
        
        pv_new = np.zeros(order)
        for i in range(order):
            for j in range(order):
                temp = q_single(xs[i], mu_array[n-1], sigma, a_prev, -b, -a_prev, b, 
                               T_curr, xs_prev[j], trunc_num, threshold, fixed_terms)
                pv_new[i] += temp * ws_pv_prev[j]
        
        xs_prev = xs.copy()
        ws_pv_prev = ws * pv_new
    
    # Final stage: compute FPTD weighted by entry distribution
    a_final = a - b * sacc_array[d-1]
    result = 0.0
    for i in range(order):
        fptd = fptd_single(t - sacc_array[d-1], mu_array[d-1], sigma, 
                          a_final, -b, -a_final, b, xs_prev[i], bdy,
                          trunc_num, threshold, fixed_terms)
        result += fptd * ws_pv_prev[i]
    
    return result


class TestMultiStageEquivalence:
    """Test equivalence of multi-stage implementations."""
    
    # Tolerances account for float32 vs float64 and minor algorithmic differences
    RTOL = 1e-5
    ATOL = 1e-6
    TRUNC_NUM = 50
    
    @pytest.fixture
    def addm_params(self):
        """Standard aDDM parameters."""
        return {
            'sigma': 1.0, 'a': 1.5, 'b': 0.3, 'x0': 0.0
        }
    
    def test_single_stage(self, addm_params):
        """Test d=1 case (single stage, no propagation)."""
        t = 1.0
        d = 1
        mu_array = np.array([0.5])
        sacc_array = np.array([0.0])
        bdy = 1
        
        # Pad arrays for JAX
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - 1))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - 1))
        
        py_result = get_addm_fptd_python(
            t, d, mu_array, sacc_array, **addm_params, bdy=bdy,
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            **addm_params, bdy=bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(py_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)
    
    @pytest.mark.parametrize("bdy", [1, -1])
    def test_two_stages(self, addm_params, bdy):
        """Test d=2 case."""
        t = 2.0
        d = 2
        mu_array = np.array([0.3, -0.2])
        sacc_array = np.array([0.0, 1.0])
        
        # Pad arrays for JAX
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - d))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - d))
        
        py_result = get_addm_fptd_python(
            t, d, mu_array, sacc_array, **addm_params, bdy=bdy,
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            **addm_params, bdy=bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(py_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)
    
    @pytest.mark.parametrize("bdy", [1, -1])
    def test_three_stages(self, addm_params, bdy):
        """Test d=3 case (multiple intermediate stages)."""
        t = 3.0
        d = 3
        mu_array = np.array([0.5, -0.3, 0.2])
        sacc_array = np.array([0.0, 1.0, 2.0])
        
        # Pad arrays
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - d))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - d))
        
        py_result = get_addm_fptd_python(
            t, d, mu_array, sacc_array, **addm_params, bdy=bdy,
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            **addm_params, bdy=bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(py_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)
    
    @pytest.mark.parametrize("bdy", [1, -1])
    def test_four_stages(self, addm_params, bdy):
        """Test d=4 case."""
        t = 4.0
        d = 4
        mu_array = np.array([0.5, -0.3, 0.2, -0.1])
        sacc_array = np.array([0.0, 1.0, 2.0, 3.0])
        
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - d))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - d))
        
        py_result = get_addm_fptd_python(
            t, d, mu_array, sacc_array, **addm_params, bdy=bdy,
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            **addm_params, bdy=bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(py_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)
    
    @pytest.mark.parametrize("t_offset", [0.1, 0.5, 0.9])
    def test_varied_response_times(self, addm_params, t_offset):
        """Test with different response times within the final stage."""
        d = 3
        mu_array = np.array([0.5, -0.3, 0.2])
        sacc_array = np.array([0.0, 1.0, 2.0])
        t = 2.0 + t_offset  # Response in final stage
        bdy = 1
        
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - d))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - d))
        
        py_result = get_addm_fptd_python(
            t, d, mu_array, sacc_array, **addm_params, bdy=bdy,
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            **addm_params, bdy=bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(py_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)
    
    @pytest.mark.parametrize("x0", [-0.3, 0.0, 0.3])
    def test_varied_starting_position(self, x0):
        """Test with different starting positions."""
        params = {'sigma': 1.0, 'a': 1.5, 'b': 0.3}
        t = 2.5
        d = 3
        mu_array = np.array([0.5, -0.3, 0.2])
        sacc_array = np.array([0.0, 1.0, 2.0])
        bdy = 1
        
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - d))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - d))
        
        py_result = get_addm_fptd_python(
            t, d, mu_array, sacc_array, **params, x0=x0, bdy=bdy,
            trunc_num=self.TRUNC_NUM, fixed_terms=True
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            **params, x0=x0, bdy=bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(py_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)


class TestBatchedMultiStage:
    """Test batched computation using vmap."""
    
    # Tolerances account for float32 vs float64 and minor algorithmic differences
    RTOL = 1e-5
    ATOL = 1e-6
    TRUNC_NUM = 50
    
    def test_vmap_over_trials(self):
        """Test vmapping over multiple trials."""
        n_trials = 5
        max_d = 4
        
        # Create test data
        rts = np.array([1.5, 2.0, 2.5, 3.0, 1.8])
        choices = np.array([1, -1, 1, -1, 1])
        lengths = np.array([2, 3, 2, 4, 3])
        
        # Padded drift arrays
        mu_arrays = np.array([
            [0.3, -0.2, 0.0, 0.0],
            [0.5, -0.3, 0.2, 0.0],
            [0.4, -0.1, 0.0, 0.0],
            [0.2, -0.4, 0.3, -0.1],
            [0.6, -0.2, 0.1, 0.0],
        ])
        
        sacc_arrays = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.8, 1.6, 0.0],
            [0.0, 1.2, 0.0, 0.0],
            [0.0, 0.7, 1.4, 2.1],
            [0.0, 0.5, 1.2, 0.0],
        ])
        
        sigma, a, b, x0 = 1.0, 1.5, 0.3, 0.0
        
        # Reference: loop over trials
        py_results = []
        for i in range(n_trials):
            result = get_addm_fptd_python(
                rts[i], int(lengths[i]), 
                mu_arrays[i, :int(lengths[i])],
                sacc_arrays[i, :int(lengths[i])],
                sigma, a, b, x0, int(choices[i]),
                trunc_num=self.TRUNC_NUM, fixed_terms=True
            )
            py_results.append(result)
        py_results = np.array(py_results)
        
        # JAX: vmap over trials
        def single_trial(rt, choice, mu_array, sacc_array, d):
            return get_addm_fptd_jax(
                rt, d, mu_array, sacc_array, 
                sigma, a, b, x0, choice, 
                trunc_num=self.TRUNC_NUM
            )
        
        batched_fn = vmap(single_trial, in_axes=(0, 0, 0, 0, 0))
        jax_results = batched_fn(
            jnp.array(rts), jnp.array(choices),
            jnp.array(mu_arrays), jnp.array(sacc_arrays),
            jnp.array(lengths)
        )
        
        np.testing.assert_allclose(py_results, np.array(jax_results), 
                                   rtol=self.RTOL, atol=self.ATOL)


class TestCythonEquivalence:
    """Test equivalence with Cython implementation (if available)."""
    
    RTOL = 1e-6  # Looser tolerance due to early termination differences
    ATOL = 1e-8
    TRUNC_NUM = 100  # More terms to match Cython behavior
    
    @pytest.fixture
    def cython_available(self):
        """Check if Cython implementation is available."""
        try:
            from efficient_fpt.multi_stage_cy import get_addm_fptd_cy
            return True
        except ImportError:
            return False
    
    @pytest.mark.parametrize("bdy", [1, -1])
    def test_against_cython(self, cython_available, bdy):
        """Compare JAX with Cython implementation."""
        if not cython_available:
            pytest.skip("Cython implementation not available")
        
        from efficient_fpt.multi_stage_cy import get_addm_fptd_cy
        
        t = 2.5
        d = 3
        mu_array = np.array([0.5, -0.3, 0.2])
        sacc_array = np.array([0.0, 1.0, 2.0])
        sigma, a, b, x0 = 1.0, 1.5, 0.3, 0.0
        
        max_d = 5
        mu_array_padded = np.pad(mu_array, (0, max_d - d))
        sacc_array_padded = np.pad(sacc_array, (0, max_d - d))
        
        cy_result = get_addm_fptd_cy(
            t, d, mu_array, sacc_array, sigma, a, b, x0, bdy,
            trunc_num=self.TRUNC_NUM, threshold=1e-30  # Minimize early termination
        )
        
        jax_result = get_addm_fptd_jax(
            t, d, jnp.array(mu_array_padded), jnp.array(sacc_array_padded), 
            sigma, a, b, x0, bdy, trunc_num=self.TRUNC_NUM
        )
        
        np.testing.assert_allclose(cy_result, float(jax_result), 
                                   rtol=self.RTOL, atol=self.ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

