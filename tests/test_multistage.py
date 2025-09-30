from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.single_stage_cy import fptd_single_cy, q_single_cy

from efficient_fpt.multi_stage import get_multistage_densities
from efficient_fpt.multi_stage_cy import get_addm_fptd_cy
from efficient_fpt.utils import check_valid_multistage_params

import numpy as np

def test_multi_stage_fptd():
    sigma = 1.0
    a = 1.5
    b = 0.3
    x0 = -0.5
    mu_array = np.array([1.0, -0.2, 1.5, 0.5, -1.0, 1.0, -0.2, 1.5, 0.5, -1.0], dtype=np.float64)
    fixation_array = np.array([0.5, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.25, 0.5], dtype=np.float64)
    sacc_array = np.cumsum(fixation_array, dtype=np.float64)
    sacc_array = np.concatenate(([0], sacc_array), dtype=np.float64)
    rt_array = (sacc_array[1:] + sacc_array[:-1]) / 2
    T = sacc_array[-1]
    sacc_array = sacc_array[:-1]
    d = len(mu_array)
    sigma_array = np.full(d, sigma, dtype=np.float64)
    a1, a2 = a, -a
    b1_array = np.full(d, -b, dtype=np.float64)
    b2_array = np.full(d, b, dtype=np.float64)
    check_valid_multistage_params(mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array)
    # Run tests
    result_cy = np.zeros(d, dtype=np.float64)

    for n in range(d):
        result_cy[n] = get_addm_fptd_cy(rt_array[n], n + 1, mu_array, sacc_array, sigma, a, b, x0, 1)
        
    result_np = get_multistage_densities(rt_array, mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array, T, np.array([[1], [x0]]))
    result_np = result_np[0][1]
    
    print(-np.log(result_np))
    print(-np.log(result_cy))

    assert np.allclose(result_cy, result_np, atol=1e-10)
    
test_multi_stage_fptd()