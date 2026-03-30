from efficient_fpt.single_stage import fptd_single, q_single
from efficient_fpt.cython.single_stage import (
    fptd_single as fptd_single_cy,
    q_single as q_single_cy,
)

from efficient_fpt.multi_stage import compute_homog_multistage_fptds_and_npd
from efficient_fpt.cython.multi_stage import compute_addm_fptd
from efficient_fpt.validation import check_multistage_params

import numpy as np


def test_single_stage_fptd():
    mu = 1.0
    sigma = 1
    x0 = -0.5

    a = 1.5
    b = 0.3
    T = 5

    ts = np.linspace(0, T, 11)[1:-1]
    result_np1 = fptd_single(
        ts, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1
    )
    result_np2 = compute_homog_multistage_fptds_and_npd(
        ts,
        T=T,
        x0=np.array([[1], [x0]]),
        a1=a,
        a2=-a,
        mu_array=np.array([mu]),
        node_array=np.array([0.0]),
        sigma_array=np.array([sigma]),
        b1_array=np.array([-b]),
        b2_array=np.array([b]),
    )
    result_np2 = result_np2[0][1]

    result_cy1 = np.zeros_like(result_np1)
    for i, t in enumerate(ts):
        result_cy1[i] = fptd_single_cy(
            t, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1
        )
    result_cy2 = np.zeros_like(result_np1)
    for i, t in enumerate(ts):
        result_cy2[i] = compute_addm_fptd(
            t, 1, 0.0, 1.0, sigma, a, b, x0, mu, 0.0, 0, np.array([0.0]), 1
        )

    assert np.allclose(result_np1, result_cy1, atol=1e-10)
    assert np.allclose(result_np1, result_np2, atol=1e-10)
    assert np.allclose(result_np1, result_cy2, atol=1e-10)


def test_cython_single_stage_supports_fixed_truncation():
    mu = 0.7
    sigma = 1.0
    x0 = -0.2
    a = 1.3
    b = 0.25
    trunc_num = 8
    threshold = 1e-3

    t = 0.9
    x = 0.15
    T = 0.9

    np_fptd = fptd_single(
        t,
        mu=mu,
        sigma=sigma,
        a1=a,
        b1=-b,
        a2=-a,
        b2=b,
        x0=x0,
        bdy=1,
        trunc_num=trunc_num,
        threshold=threshold,
        adaptive_stopping=False,
    )
    cy_fptd = fptd_single_cy(
        t,
        mu=mu,
        sigma=sigma,
        a1=a,
        b1=-b,
        a2=-a,
        b2=b,
        x0=x0,
        bdy=1,
        trunc_num=trunc_num,
        threshold=threshold,
        adaptive_stopping=False,
    )

    np_q = q_single(
        x,
        mu=mu,
        sigma=sigma,
        a1=a,
        b1=-b,
        a2=-a,
        b2=b,
        T=T,
        x0=x0,
        trunc_num=trunc_num,
        threshold=threshold,
        adaptive_stopping=False,
    )
    cy_q = q_single_cy(
        x,
        mu=mu,
        sigma=sigma,
        a1=a,
        b1=-b,
        a2=-a,
        b2=b,
        T=T,
        x0=x0,
        trunc_num=trunc_num,
        threshold=threshold,
        adaptive_stopping=False,
    )

    assert np.allclose(cy_fptd, np_fptd, atol=1e-10)
    assert np.allclose(cy_q, np_q, atol=1e-10)
