from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.cython.single_stage import (
    fptd_single as fptd_single_cy,
    q_single as q_single_cy,
)

from efficient_fpt.multi_stage import compute_homog_multistage_fptds_and_npd
from efficient_fpt.cython.multi_stage import compute_heterog_multistage_fptd
from efficient_fpt.validation import check_multistage_params

import numpy as np


def test_multi_stage_fptd():
    """Cython FPTD matches Python reference for a 10-stage problem."""
    sigma = 1.0
    a = 1.5
    b = 0.3
    x0 = -0.5
    mu_array = np.array(
        [1.0, -0.2, 1.5, 0.5, -1.0, 1.0, -0.2, 1.5, 0.5, -1.0], dtype=np.float64
    )
    fixation_array = np.array(
        [0.5, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.25, 0.5], dtype=np.float64
    )
    node_array = np.cumsum(fixation_array, dtype=np.float64)
    node_array = np.concatenate(([0], node_array), dtype=np.float64)
    rt_array = (node_array[1:] + node_array[:-1]) / 2
    T = node_array[-1]
    node_array = node_array[:-1]
    d = len(mu_array)
    sigma_array = np.full(d, sigma, dtype=np.float64)
    a1, a2 = a, -a
    b1_array = np.full(d, -b, dtype=np.float64)
    b2_array = np.full(d, b, dtype=np.float64)
    check_multistage_params(
        mu_array, node_array, sigma_array, a1, b1_array, a2, b2_array
    )

    # Cython path: evaluate FPTD one stage at a time
    result_cy = np.zeros(d, dtype=np.float64)
    for n in range(d):
        result_cy[n] = compute_heterog_multistage_fptd(
            rt_array[n],
            1,
            x0,
            a1,
            a2,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            n + 1,
        )

    # Python path: vectorised multistage density
    result_np = compute_homog_multistage_fptds_and_npd(
        rt_array,
        T,
        np.array([[1], [x0]]),
        a1,
        a2,
        mu_array,
        node_array,
        sigma_array,
        b1_array,
        b2_array,
    )
    result_np = result_np[0][1]

    assert np.allclose(result_cy, result_np, atol=1e-10)
