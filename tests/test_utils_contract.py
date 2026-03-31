import math

import numpy as np
import pytest

from efficient_fpt.numpy.utils import positive_log as np_positive_log


def test_numpy_positive_log_contract():
    values = np.array([2.0, 1.0, 0.0, -3.0, np.nan], dtype=np.float64)
    out = np_positive_log(values)

    assert np.isclose(out[0], math.log(2.0))
    assert np.isclose(out[1], 0.0)
    assert np.isneginf(out[2])
    assert np.isneginf(out[3])
    assert np.isnan(out[4])


def test_cython_positive_log_contract():
    cy_utils = pytest.importorskip("efficient_fpt.cython.utils")
    cy_positive_log = cy_utils.positive_log_wrapper
    assert math.isclose(cy_positive_log(2.0), math.log(2.0))
    assert math.isclose(cy_positive_log(1.0), 0.0)
    assert math.isinf(cy_positive_log(0.0)) and cy_positive_log(0.0) < 0.0
    assert math.isinf(cy_positive_log(-3.0)) and cy_positive_log(-3.0) < 0.0
    assert math.isnan(cy_positive_log(float("nan")))


def test_jax_positive_log_contract():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from efficient_fpt.jax.utils import (
        get_jax_dtype,
        positive_log as jax_positive_log,
    )

    values = jnp.array([2.0, 1.0, 0.0, -3.0, jnp.nan], dtype=get_jax_dtype())
    out = np.asarray(jax_positive_log(values))

    assert np.isclose(out[0], math.log(2.0))
    assert np.isclose(out[1], 0.0)
    assert np.isneginf(out[2])
    assert np.isneginf(out[3])
    assert np.isnan(out[4])
