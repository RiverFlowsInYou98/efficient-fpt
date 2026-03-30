"""Public naming, signature, and schema regression tests."""

import inspect
import pickle
import subprocess
import sys

import numpy as np
import pytest

import efficient_fpt.jax as jax_api
from efficient_fpt.numpy.single_stage import (
    fptd_single as np_fptd_single,
    q_single as np_q_single,
)
from efficient_fpt.jax.single_stage import (
    fptd_single as jax_fptd_single,
    q_single as jax_q_single,
)
from efficient_fpt import (
    aDDModel,
    load_addm_experiment,
    save_addm_experiment,
)
from efficient_fpt.io import (
    _GROUP_KEYS,
    _DECISION_KEYS,
    _PARAM_KEYS,
    _COVARIATE_KEYS,
    _CONFIG_KEYS,
)
from efficient_fpt.jax.batch import (
    compute_addm_likelihoods_batchvmap,
    compute_addm_likelihoods_batchscan,
    compute_addm_likelihoods,
    make_addm_nll_function_batchvmap,
    make_addm_nll_function_batchscan,
    compute_addm_nll,
    make_addm_nll_function,
)
from efficient_fpt.jax.multi_stage import (
    compute_addm_fptd,
    compute_addm_fptd_precomputed,
    compute_addm_fptd_stagescan,
    compute_heterog_multistage_fptd,
    compute_heterog_multistage_fptd_precomputed,
    compute_heterog_multistage_fptd_stagescan,
)
from efficient_fpt.multi_stage import compute_homog_multistage_fptds_and_npd

try:
    import efficient_fpt.cython as cython_api
except ImportError:  # pragma: no cover - exercised in rebuilt environment
    cython_api = None


def _parameter_names(fn):
    return list(inspect.signature(fn).parameters)


def _keyword_only_names(fn):
    return [
        name
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def _run_python_snippet(snippet):
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()


def _pack_generated_experiment(data):
    return {
        "decision_data": {
            "rt_data": data["rt_data"],
            "choice_data": data["choice_data"],
        },
        "params": {
            key: data["params"][key] for key in sorted(_PARAM_KEYS)
        },
        "covariates": {
            "r1_data": data["r1_data"],
            "r2_data": data["r2_data"],
            "flag_data": data["flag_data"],
            "sacc_array_data": data["sacc_array_data"],
            "d_data": data["d_data"],
        },
        "config": {
            key: data["params"][key] for key in sorted(_CONFIG_KEYS)
        },
    }


def test_public_signatures_follow_canonical_order():
    assert compute_addm_fptd is compute_addm_fptd_precomputed
    assert compute_heterog_multistage_fptd is (
        compute_heterog_multistage_fptd_precomputed
    )
    assert compute_addm_likelihoods is compute_addm_likelihoods_batchscan
    assert make_addm_nll_function is make_addm_nll_function_batchscan
    assert _parameter_names(compute_addm_fptd) == [
        "rt",
        "choice",
        "eta",
        "kappa",
        "sigma",
        "a",
        "b",
        "x0",
        "r1",
        "r2",
        "flag",
        "sacc_array",
        "d",
        "order",
        "trunc_num",
        "log_space",
    ]


def test_importing_efficient_fpt_jax_does_not_change_x64_flag():
    lines = _run_python_snippet(
        "import jax; "
        "before = jax.config.read('jax_enable_x64'); "
        "import efficient_fpt.jax; "
        "after = jax.config.read('jax_enable_x64'); "
        "print(before); print(after)"
    )
    assert lines[0] == lines[1]


def test_set_jax_precision_explicitly_controls_x64_flag():
    lines = _run_python_snippet(
        "import jax; "
        "from efficient_fpt.jax.utils import set_jax_precision; "
        "print(jax.config.read('jax_enable_x64')); "
        "set_jax_precision(True); "
        "print(jax.config.read('jax_enable_x64')); "
        "set_jax_precision(False); "
        "print(jax.config.read('jax_enable_x64'))"
    )
    assert lines == ["False", "True", "False"]
    assert _parameter_names(compute_addm_fptd_precomputed) == _parameter_names(
        compute_addm_fptd
    )
    assert _parameter_names(compute_addm_fptd_stagescan) == [
        "rt",
        "choice",
        "eta",
        "kappa",
        "sigma",
        "a",
        "b",
        "x0",
        "r1",
        "r2",
        "flag",
        "sacc_array",
        "d",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _parameter_names(compute_addm_likelihoods) == [
        "rt_data",
        "choice_data",
        "eta",
        "kappa",
        "sigma",
        "a",
        "b",
        "x0",
        "r1_data",
        "r2_data",
        "flag_data",
        "sacc_array_data",
        "d_data",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ]
    assert _parameter_names(compute_addm_likelihoods_batchscan) == _parameter_names(
        compute_addm_likelihoods
    )
    assert _parameter_names(compute_addm_likelihoods_batchvmap) == _parameter_names(
        compute_addm_likelihoods
    )
    assert _parameter_names(compute_addm_nll) == _parameter_names(
        compute_addm_likelihoods
    ) + ["reduce", "warn"]
    assert _parameter_names(make_addm_nll_function) == [
        "rt_data",
        "choice_data",
        "r1_data",
        "r2_data",
        "flag_data",
        "sacc_array_data",
        "d_data",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ]
    assert _parameter_names(make_addm_nll_function_batchscan) == _parameter_names(
        make_addm_nll_function
    )
    assert _parameter_names(make_addm_nll_function_batchvmap) == _parameter_names(
        make_addm_nll_function
    )
    assert _parameter_names(aDDModel.mean_neg_log_likelihood) == [
        "self",
        "rt_data",
        "choice_data",
        "r1_data",
        "r2_data",
        "flag_data",
        "sacc_array_data",
        "d_data",
        "order",
        "trunc_num",
        "threshold",
        "n_threads",
        "log_space",
        "warn",
    ]
    assert _parameter_names(compute_heterog_multistage_fptd) == [
        "rt",
        "choice",
        "x0",
        "a1",
        "a2",
        "mu_array",
        "node_array",
        "sigma_array",
        "b1_array",
        "b2_array",
        "d",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _parameter_names(
        compute_heterog_multistage_fptd_precomputed
    ) == _parameter_names(compute_heterog_multistage_fptd)
    assert _parameter_names(
        compute_heterog_multistage_fptd_stagescan
    ) == _parameter_names(compute_heterog_multistage_fptd)
    assert _parameter_names(compute_homog_multistage_fptds_and_npd) == [
        "t_grid",
        "T",
        "x0",
        "a1",
        "a2",
        "mu_array",
        "node_array",
        "sigma_array",
        "b1_array",
        "b2_array",
        "order",
        "eps",
        "trunc_num",
        "threshold",
        "adaptive_stopping",
        "log_space",
    ]


def test_compute_parameters_are_keyword_only_on_public_fptd_apis():
    assert _keyword_only_names(np_fptd_single) == [
        "trunc_num",
        "threshold",
        "adaptive_stopping",
    ]
    assert _keyword_only_names(np_q_single) == [
        "trunc_num",
        "threshold",
        "adaptive_stopping",
    ]
    assert _keyword_only_names(jax_fptd_single) == ["trunc_num"]
    assert _keyword_only_names(jax_q_single) == ["trunc_num"]
    assert _keyword_only_names(compute_addm_fptd) == [
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _keyword_only_names(compute_addm_fptd_precomputed) == _keyword_only_names(
        compute_addm_fptd
    )
    assert _keyword_only_names(compute_addm_fptd_stagescan) == [
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _keyword_only_names(compute_heterog_multistage_fptd) == [
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _keyword_only_names(
        compute_heterog_multistage_fptd_precomputed
    ) == _keyword_only_names(compute_heterog_multistage_fptd)
    assert _keyword_only_names(
        compute_heterog_multistage_fptd_stagescan
    ) == _keyword_only_names(compute_heterog_multistage_fptd)
    assert _keyword_only_names(compute_addm_likelihoods) == [
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ]
    assert _keyword_only_names(
        compute_addm_likelihoods_batchscan
    ) == _keyword_only_names(compute_addm_likelihoods)
    assert _keyword_only_names(
        compute_addm_likelihoods_batchvmap
    ) == _keyword_only_names(compute_addm_likelihoods)
    assert _keyword_only_names(compute_addm_nll) == [
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
        "reduce",
        "warn",
    ]
    assert _keyword_only_names(make_addm_nll_function) == [
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ]
    assert _keyword_only_names(
        make_addm_nll_function_batchscan
    ) == _keyword_only_names(make_addm_nll_function)
    assert _keyword_only_names(
        make_addm_nll_function_batchvmap
    ) == _keyword_only_names(make_addm_nll_function)
    assert _keyword_only_names(compute_homog_multistage_fptds_and_npd) == [
        "order",
        "eps",
        "trunc_num",
        "threshold",
        "adaptive_stopping",
        "log_space",
    ]

    if cython_api is not None:
        assert _keyword_only_names(cython_api.fptd_single) == []
        assert _keyword_only_names(cython_api.q_single) == []
        assert _keyword_only_names(cython_api.compute_addm_fptd) == []
        assert _keyword_only_names(cython_api.compute_heterog_multistage_fptd) == []


def test_removed_public_aliases_are_absent():
    banned = {
        "get_multistage_densities",
        "get_multistage_fptd",
        "get_multistage_fptd_old",
        "get_addm_fptd",
        "get_addm_fptd_old",
        "get_heterog_multistage_fptd",
        "get_homog_multistage_densities",
        "compute_likelihoods_batch",
        "compute_nll_batch",
        "compute_nll_batch_sum",
        "make_nll_function",
        "compute_llhds_serial",
        "compute_loss_serial",
        "compute_loss_parallel",
        "compute_tadaloss_parallel",
        "pad_sacc_array_safely",
        "safe_sacc",
        "length_data",
        "generate_addm_experiment",
    }
    # These are only removed from the JAX API (Cython keeps them)
    jax_only_banned = {"compute_addm_mean_nll", "compute_addm_sum_nll"}
    for name in banned:
        assert not hasattr(jax_api, name)
        if cython_api is not None:
            assert not hasattr(cython_api, name)
    for name in jax_only_banned:
        assert not hasattr(jax_api, name)


def test_generate_experiment_returns_flat_simulation_schema():
    model = aDDModel(eta=0.2, kappa=1.0, sigma=1.0, a=1.5, b=0.2, x0=0.0)
    data = model.generate_experiment(
        n_trials=4,
        dt=0.01,
        T=1.5,
        gamma_shape=2.0,
        gamma_scale=0.2,
        r_range=(0.0, 1.0),
        random_state=0,
    )

    assert set(data) == _DECISION_KEYS | _COVARIATE_KEYS | {"params"}
    assert "mu1_data" not in data
    assert "mu2_data" not in data
    assert "mu_array_data" not in data


def test_save_and_load_addm_experiment_roundtrip_keeps_grouped_keys(tmp_path):
    model = aDDModel(eta=0.1, kappa=1.2, sigma=1.0, a=1.4, b=0.25, x0=0.0)
    flat = model.generate_experiment(
        n_trials=3,
        dt=0.01,
        T=1.5,
        gamma_shape=2.0,
        gamma_scale=0.2,
        r_range=(0.0, 1.0),
        random_state=1,
    )
    data = _pack_generated_experiment(flat)
    data["note"] = "preserve me"
    data["params"]["custom_prior"] = {"eta": "beta"}
    data["covariates"]["mu1_data"] = np.array([0.1, 0.2, 0.3])

    path = tmp_path / "addm.pkl"
    save_addm_experiment(path, data)

    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    assert set(raw) == _GROUP_KEYS | {"note"}
    assert raw["note"] == "preserve me"
    assert raw["params"]["custom_prior"] == {"eta": "beta"}
    np.testing.assert_array_equal(raw["covariates"]["mu1_data"], data["covariates"]["mu1_data"])

    loaded = load_addm_experiment(path)
    assert set(loaded) == _GROUP_KEYS | {"note"}
    for key in _DECISION_KEYS:
        np.testing.assert_array_equal(loaded["decision_data"][key], data["decision_data"][key])
    for key in _COVARIATE_KEYS:
        np.testing.assert_array_equal(loaded["covariates"][key], data["covariates"][key])
    assert loaded["params"] == data["params"]
    assert loaded["config"] == data["config"]
    assert loaded["note"] == "preserve me"


def test_save_addm_experiment_rejects_incomplete_canonical_payload(tmp_path):
    path = tmp_path / "bad_addm.pkl"
    incomplete = {
        "decision_data": {
            "rt_data": np.array([1.0]),
            "choice_data": np.array([1], dtype=np.int32),
        },
        "params": {"a": 1.0},
    }

    with pytest.raises(ValueError, match="Missing required groups"):
        save_addm_experiment(path, incomplete)

    incomplete_groups = {
        "decision_data": {
            "rt_data": np.array([1.0]),
            "choice_data": np.array([1], dtype=np.int32),
        },
        "covariates": {
            "r1_data": np.array([1.0]),
            "r2_data": np.array([2.0]),
            "flag_data": np.array([0], dtype=np.int32),
            "sacc_array_data": np.array([[0.0]]),
            "d_data": np.array([1], dtype=np.int32),
        },
        "params": {"a": 1.0},
        "config": {"T": 1.0},
    }
    with pytest.raises(ValueError, match="Missing required keys in 'params'"):
        save_addm_experiment(path, incomplete_groups)

    bad_lengths = {
        "decision_data": {
            "rt_data": np.array([1.0]),
            "choice_data": np.array([1], dtype=np.int32),
        },
        "covariates": {
            "r1_data": np.array([1.0, 2.0]),
            "r2_data": np.array([2.0, 3.0]),
            "flag_data": np.array([0, 1], dtype=np.int32),
            "sacc_array_data": np.array([[0.0], [0.0]]),
            "d_data": np.array([1, 1], dtype=np.int32),
        },
        "params": {key: 1.0 for key in _PARAM_KEYS},
        "config": {
            "dt": 0.01,
            "T": 1.0,
            "gamma_shape": 2.0,
            "gamma_scale": 0.3,
            "r_range": (0.0, 1.0),
        },
    }
    with pytest.raises(ValueError, match="expected 1"):
        save_addm_experiment(path, bad_lengths)


def test_load_addm_experiment_rejects_legacy_payload(tmp_path):
    legacy = {
        "rt_data": np.array([1.2, 1.7], dtype=np.float64),
        "choice_data": np.array([1, -1], dtype=np.int32),
        "r1_data": np.array([0.4, 0.2], dtype=np.float64),
        "r2_data": np.array([0.3, 0.5], dtype=np.float64),
        "flag_data": np.array([0, 1], dtype=np.int32),
        "sacc_array_data": np.array([[0.0, 0.0], [0.0, 0.7]], dtype=np.float64),
        "d_data": np.array([1, 2], dtype=np.int32),
        "params": {"eta": 0.1},
    }
    path = tmp_path / "legacy_addm.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle)

    with pytest.raises(ValueError, match="Missing required groups"):
        load_addm_experiment(path)
