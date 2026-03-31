import numpy as np
import pytest

pytest.importorskip("efficient_fpt.cython.single_stage")

from efficient_fpt.single_stage import fptd_single, log_fptd_single, q_single
from efficient_fpt.cython.single_stage import (
    fptd_single as fptd_single_cy,
    log_fptd_single as log_fptd_single_cy,
    q_single as q_single_cy,
)
from efficient_fpt.multi_stage import compute_homog_multistage_logfptds_and_lognpd
from efficient_fpt.cython.multi_stage import compute_addm_logfptd
from efficient_fpt.validation import check_multistage_params


def test_single_stage_fptd():
    mu = 1.0
    sigma = 1.0
    x0 = -0.5
    a = 1.5
    b = 0.3
    T = 5.0

    ts = np.linspace(0.0, T, 11)[1:-1]
    result_np1 = fptd_single(
        ts, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1
    )
    result_np2, _ = compute_homog_multistage_logfptds_and_lognpd(
        ts,
        T=T,
        x0=np.array([[1.0], [x0]]),
        a1=a,
        a2=-a,
        mu_array=np.array([mu]),
        node_array=np.array([0.0]),
        sigma_array=np.array([sigma]),
        b1_array=np.array([-b]),
        b2_array=np.array([b]),
    )
    result_np2 = np.exp(result_np2[1])

    result_cy1 = np.zeros_like(result_np1)
    result_cy2 = np.zeros_like(result_np1)
    result_log_cy = np.zeros_like(result_np1)
    for i, t in enumerate(ts):
        result_cy1[i] = fptd_single_cy(
            t, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1
        )
        result_cy2[i] = np.exp(
            compute_addm_logfptd(
                t, 1, 0.0, 1.0, sigma, a, b, x0, mu, 0.0, 0, np.array([0.0]), 1
            )
        )
        result_log_cy[i] = log_fptd_single_cy(
            t, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1
        )

    np.testing.assert_allclose(result_np1, result_cy1, atol=1e-10)
    np.testing.assert_allclose(result_np1, result_np2, atol=1e-10)
    np.testing.assert_allclose(result_np1, result_cy2, atol=1e-10)
    np.testing.assert_allclose(np.exp(result_log_cy), result_np1, atol=1e-10)
    np.testing.assert_allclose(np.exp(log_fptd_single(ts, mu=mu, sigma=sigma, a1=a, b1=-b, a2=-a, b2=b, x0=x0, bdy=1)), result_np1, atol=1e-10)


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


def test_check_multistage_params_accepts_single_stage_schedule():
    check_multistage_params(
        np.array([0.2]),
        np.array([0.0]),
        np.array([1.0]),
        1.0,
        np.array([-0.2]),
        -1.0,
        np.array([0.2]),
    )
