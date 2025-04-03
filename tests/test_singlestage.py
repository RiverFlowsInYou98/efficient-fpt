from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.single_stage_cy import fptd_single_cy, q_single_cy

from efficient_fpt.multi_stage import get_multistage_densities
from efficient_fpt.multi_stage_cy import get_addm_fptd_cy
from efficient_fpt.utils import check_valid_multistage_params

import numpy as np

def test_single_stage_fptd():
    mu = 1.0
    sigma = 1
    x0 = -0.5

    a = 1.5
    b = 0.3
    T = 5

    ts = np.linspace(0, T, 11)[1:-1]
    result_np1 = fptd_single(ts, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1)
    result_np2 = get_multistage_densities(ts, mu_array=np.array([mu]), sacc_array=np.array([0.0]), sigma_array=np.array([sigma]), a1=a, b1_array=np.array([-b]), a2=-a, b2_array=np.array([b]), T=T, x0=np.array([[1], [x0]]))
    result_np2 = result_np2[0][1]

    result_cy1 = np.zeros_like(result_np1)
    for i, t in enumerate(ts):
        result_cy1[i] = fptd_single_cy(t, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1)
    result_cy2 = np.zeros_like(result_np1)
    for i, t in enumerate(ts):
        result_cy2[i] = get_addm_fptd_cy(t, d=1, mu_array=np.array([mu]), sacc_array=np.array([0.0]), sigma=sigma, a=a, b=b, x0=x0, bdy=1)
    
    assert np.allclose(result_np1, result_cy1, atol=1e-10)
    assert np.allclose(result_np1, result_np2, atol=1e-10)
    assert np.allclose(result_np1, result_cy2, atol=1e-10)