"""Public naming, signature, and schema regression tests."""

from __future__ import annotations

import inspect
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import efpt.multi_stage as top_level_multi_stage
import efpt.single_stage as top_level_single_stage
import efpt.jax as jax_api
from efpt import (
    aDDModel,
    load_addm_experiment,
    load_simulation,
    save_addm_experiment,
    save_simulation,
)
from efpt.io import (
    _CONFIG_KEYS,
    _COVARIATE_KEYS,
    _DECISION_KEYS,
    _GROUP_KEYS,
    _PARAM_KEYS,
)
from efpt.jax.batch import (
    compute_addm_loglikelihoods,
    compute_addm_loglikelihoods_batchscan,
    compute_addm_loglikelihoods_batchvmap,
    compute_addm_nll,
    make_addm_nll_function,
    make_addm_nll_function_batchscan,
    make_addm_nll_function_batchvmap,
)
from efpt.jax.multi_stage import (
    compute_addm_logfptd,
    compute_addm_logfptd_precomputed,
    compute_addm_logfptd_stagescan,
    compute_heterog_multistage_logfptd,
    compute_heterog_multistage_logfptd_precomputed,
    compute_heterog_multistage_logfptd_stagescan,
)
from efpt.jax.single_stage import (
    fptd_single as jax_fptd_single,
    log_fptd_single as jax_log_fptd_single,
    log_q_single as jax_log_q_single,
    q_single as jax_q_single,
)
from efpt.multi_stage import compute_homog_multistage_logfptds_and_lognpd
from efpt.numpy.single_stage import (
    fptd_single as np_fptd_single,
    log_fptd_single as np_log_fptd_single,
    log_q_single as np_log_q_single,
    q_single as np_q_single,
)

try:
    import efpt.cython as cython_api
except ImportError:  # pragma: no cover
    cython_api = None


def _parameter_names(fn):
    return list(inspect.signature(fn).parameters)


def _keyword_only_names(fn):
    return [
        name
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def _run_python_snippet(snippet: str):
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()


def test_public_aliases_follow_log_first_contract():
    assert compute_addm_logfptd is compute_addm_logfptd_precomputed
    assert (
        compute_heterog_multistage_logfptd
        is compute_heterog_multistage_logfptd_precomputed
    )
    assert compute_addm_loglikelihoods is compute_addm_loglikelihoods_batchvmap
    assert make_addm_nll_function is make_addm_nll_function_batchvmap


def test_cython_public_namespace_does_not_export_private_simulator_helper():
    if cython_api is None:
        pytest.skip("Cython backend not available")
    assert not hasattr(cython_api, "_simulate_addm_fpt")


def test_top_level_compatibility_modules_do_not_leak_backend_internals():
    assert not hasattr(top_level_single_stage, "np")
    assert not hasattr(top_level_single_stage, "positive_log")
    assert not hasattr(top_level_multi_stage, "np")
    assert not hasattr(top_level_multi_stage, "positive_log")
    assert not hasattr(top_level_multi_stage, "filter_and_group")


def test_importing_efpt_jax_does_not_change_x64_flag():
    lines = _run_python_snippet(
        "import jax; "
        "before = jax.config.read('jax_enable_x64'); "
        "import efpt.jax; "
        "after = jax.config.read('jax_enable_x64'); "
        "print(before); print(after)"
    )
    assert lines[0] == lines[1]


def test_set_jax_precision_explicitly_controls_x64_flag():
    lines = _run_python_snippet(
        "import jax; "
        "from efpt.jax.utils import set_jax_precision; "
        "print(jax.config.read('jax_enable_x64')); "
        "set_jax_precision(True); "
        "print(jax.config.read('jax_enable_x64')); "
        "set_jax_precision(False); "
        "print(jax.config.read('jax_enable_x64'))"
    )
    assert lines == ["False", "True", "False"]


def test_public_signatures_follow_canonical_order():
    assert _parameter_names(compute_addm_logfptd) == [
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
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _parameter_names(compute_addm_logfptd_precomputed) == _parameter_names(
        compute_addm_logfptd
    )
    assert _parameter_names(compute_addm_logfptd_stagescan) == _parameter_names(
        compute_addm_logfptd
    )
    assert _parameter_names(compute_heterog_multistage_logfptd) == [
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
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _parameter_names(
        compute_heterog_multistage_logfptd_precomputed
    ) == _parameter_names(compute_heterog_multistage_logfptd)
    assert _parameter_names(
        compute_heterog_multistage_logfptd_stagescan
    ) == _parameter_names(compute_heterog_multistage_logfptd)
    assert _parameter_names(compute_addm_loglikelihoods) == [
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
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ]
    assert _parameter_names(compute_addm_loglikelihoods_batchscan) == _parameter_names(
        compute_addm_loglikelihoods
    )
    assert _parameter_names(compute_addm_loglikelihoods_batchvmap) == _parameter_names(
        compute_addm_loglikelihoods
    )
    assert _parameter_names(compute_addm_nll) == _parameter_names(
        compute_addm_loglikelihoods
    ) + ["reduce", "invalid_policy", "warn"]
    assert _parameter_names(make_addm_nll_function) == [
        "rt_data",
        "choice_data",
        "r1_data",
        "r2_data",
        "flag_data",
        "sacc_array_data",
        "d_data",
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
        "invalid_policy",
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
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "threshold",
        "n_threads",
        "log_space",
        "invalid_policy",
        "warn",
    ]
    assert _parameter_names(compute_homog_multistage_logfptds_and_lognpd) == [
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
        "order_mid",
        "order_last",
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
    assert _keyword_only_names(np_log_fptd_single) == [
        "trunc_num",
        "threshold",
        "adaptive_stopping",
    ]
    assert _keyword_only_names(np_log_q_single) == [
        "trunc_num",
        "threshold",
        "adaptive_stopping",
    ]
    assert _keyword_only_names(jax_fptd_single) == ["trunc_num"]
    assert _keyword_only_names(jax_q_single) == ["trunc_num"]
    assert _keyword_only_names(jax_log_fptd_single) == ["trunc_num"]
    assert _keyword_only_names(jax_log_q_single) == ["trunc_num"]
    assert _keyword_only_names(compute_addm_logfptd) == [
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _keyword_only_names(compute_addm_logfptd_stagescan) == [
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _keyword_only_names(compute_heterog_multistage_logfptd) == [
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
    ]
    assert _keyword_only_names(compute_addm_loglikelihoods) == [
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ]
    assert _keyword_only_names(compute_addm_nll) == [
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
        "reduce",
        "invalid_policy",
        "warn",
    ]
    assert _keyword_only_names(make_addm_nll_function) == [
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
        "invalid_policy",
    ]
    assert _keyword_only_names(compute_homog_multistage_logfptds_and_lognpd) == [
        "order_mid",
        "order_last",
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
        assert _keyword_only_names(cython_api.log_fptd_single) == []
        assert _keyword_only_names(cython_api.log_q_single) == []
        assert _keyword_only_names(cython_api.compute_addm_logfptd) == []
        assert _keyword_only_names(cython_api.compute_heterog_multistage_logfptd) == []


def test_public_quadrature_alias_requires_old_or_new_style_not_both():
    with pytest.raises(ValueError, match="legacy order or split order_mid/order_last"):
        compute_addm_logfptd(
            rt=1.2,
            choice=1,
            eta=0.2,
            kappa=1.0,
            sigma=1.0,
            a=1.5,
            b=0.2,
            x0=0.0,
            r1=1.0,
            r2=0.5,
            flag=0,
            sacc_array=np.array([0.0, 0.5]),
            d=2,
            order_mid=24,
            order=30,
        )


def test_generalized_precomputed_order_alias_matches_split_order_arguments():
    rt = 1.2
    choice = 1
    x0 = 0.0
    a1 = 1.5
    a2 = -1.5
    mu_array = jnp.array([0.4, -0.2, 0.1], dtype=jnp.float64)
    node_array = jnp.array([0.0, 0.6, 1.0], dtype=jnp.float64)
    sigma_array = jnp.ones(3, dtype=jnp.float64)
    b1_array = jnp.full(3, -0.2, dtype=jnp.float64)
    b2_array = jnp.full(3, 0.2, dtype=jnp.float64)

    legacy = compute_heterog_multistage_logfptd_precomputed(
        rt,
        choice,
        x0,
        a1,
        a2,
        mu_array,
        node_array,
        sigma_array,
        b1_array,
        b2_array,
        3,
        order=24,
        trunc_num=6,
    )
    split = compute_heterog_multistage_logfptd_precomputed(
        rt,
        choice,
        x0,
        a1,
        a2,
        mu_array,
        node_array,
        sigma_array,
        b1_array,
        b2_array,
        3,
        order_mid=24,
        order_last=24,
        trunc_num=6,
    )
    assert float(legacy) == pytest.approx(float(split), rel=1e-12, abs=1e-12)


def test_removed_normal_space_multistage_and_batch_aliases_are_absent():
    banned = {
        "compute_addm_fptd",
        "compute_addm_fptd_precomputed",
        "compute_addm_fptd_stagescan",
        "compute_heterog_multistage_fptd",
        "compute_heterog_multistage_fptd_precomputed",
        "compute_heterog_multistage_fptd_stagescan",
        "compute_addm_likelihoods",
        "compute_addm_likelihoods_batchscan",
        "compute_addm_likelihoods_batchvmap",
        "compute_tada_likelihoods",
        "compute_homog_multistage_fptds_and_npd",
    }
    for name in banned:
        assert not hasattr(jax_api, name)
        if cython_api is not None:
            assert not hasattr(cython_api, name)


def test_generate_experiment_returns_grouped_simulation_schema():
    model = aDDModel(eta=0.2, kappa=1.0, sigma=1.0, a=1.5, b=0.2, x0=0.0)
    data = model.generate_experiment(
        n_trials=4,
        dt=0.01,
        T=1.5,
        gamma_shape=2.0,
        gamma_scale=0.2,
        r_range=(0.0, 1.0),
        rng=0,
    )

    assert set(data) == _GROUP_KEYS
    assert set(data["decision_data"]) == _DECISION_KEYS
    assert set(data["covariates"]) == _COVARIATE_KEYS
    assert set(data["params"]) == _PARAM_KEYS
    assert set(data["config"]) == _CONFIG_KEYS
    assert "mu1_data" not in data["covariates"]
    assert "mu2_data" not in data["covariates"]
    assert "mu_array_data" not in data["covariates"]


def test_save_and_load_addm_experiment_roundtrip_uses_strict_npz_schema(tmp_path):
    model = aDDModel(eta=0.1, kappa=1.2, sigma=1.0, a=1.4, b=0.25, x0=0.0)
    data = model.generate_experiment(
        n_trials=3,
        dt=0.01,
        T=1.5,
        gamma_shape=2.0,
        gamma_scale=0.2,
        r_range=(0.0, 1.0),
        rng=1,
    )

    path = tmp_path / "addm.npz"
    save_addm_experiment(path, data)

    with np.load(path, allow_pickle=False) as archive:
        assert archive["__format__"].item() == "efpt.addm_experiment"
        assert int(archive["__version__"].item()) == 1
        assert set(archive.files) == {
            "__format__",
            "__version__",
            "decision_data/rt_data",
            "decision_data/choice_data",
            "params/eta",
            "params/kappa",
            "params/sigma",
            "params/a",
            "params/b",
            "params/x0",
            "covariates/r1_data",
            "covariates/r2_data",
            "covariates/flag_data",
            "covariates/sacc_array_data",
            "covariates/d_data",
            "config/dt",
            "config/T",
            "config/gamma_shape",
            "config/gamma_scale",
            "config/r_range",
        }

    loaded = load_addm_experiment(path)
    assert set(loaded) == _GROUP_KEYS
    for key in _DECISION_KEYS:
        np.testing.assert_array_equal(
            loaded["decision_data"][key], data["decision_data"][key]
        )
    for key in _COVARIATE_KEYS:
        np.testing.assert_array_equal(
            loaded["covariates"][key], data["covariates"][key]
        )
    assert loaded["params"] == data["params"]
    assert loaded["config"] == data["config"]


def test_save_addm_experiment_rejects_incomplete_canonical_payload(tmp_path):
    path = tmp_path / "bad_addm.npz"
    incomplete = {
        "decision_data": {
            "rt_data": np.array([1.0]),
            "choice_data": np.array([1], dtype=np.int32),
        },
        "params": {"a": 1.0},
    }

    with pytest.raises(ValueError, match="Missing required keys in 'payload'"):
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


def test_save_addm_experiment_rejects_extra_groups_and_keys(tmp_path):
    path = tmp_path / "extra_addm.npz"
    model = aDDModel(eta=0.2, kappa=1.0, sigma=1.0, a=1.5, b=0.2, x0=0.0)
    data = model.generate_experiment(
        n_trials=2,
        dt=0.01,
        T=1.0,
        gamma_shape=2.0,
        gamma_scale=0.2,
        r_range=(0.0, 1.0),
        rng=0,
    )

    with_note = dict(data)
    with_note["note"] = "extra"
    with pytest.raises(ValueError, match="Unexpected keys in 'payload'"):
        save_addm_experiment(path, with_note)

    with_extra_covariate = {
        **data,
        "covariates": dict(data["covariates"], mu1_data=np.array([0.1, 0.2])),
    }
    with pytest.raises(ValueError, match="Unexpected keys in 'covariates'"):
        save_addm_experiment(path, with_extra_covariate)


def test_load_addm_experiment_rejects_non_npz_paths():
    with pytest.raises(ValueError, match=r"only supports '.npz' files"):
        load_addm_experiment("legacy_addm.pkl")


def test_load_addm_experiment_rejects_bad_archive_metadata(tmp_path):
    path = tmp_path / "bad_addm.npz"
    np.savez_compressed(
        path,
        __format__=np.asarray("wrong.format"),
        __version__=np.asarray(1, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Archive format mismatch"):
        load_addm_experiment(path)


def test_load_addm_experiment_rejects_noncanonical_archive_keys(tmp_path):
    path = tmp_path / "legacy_addm.npz"
    legacy = {
        "__format__": np.asarray("efpt.addm_experiment"),
        "__version__": np.asarray(1, dtype=np.int64),
        "rt_data": np.array([1.2, 1.7], dtype=np.float64),
    }
    np.savez_compressed(path, **legacy)

    with pytest.raises(ValueError, match="missing required experiment keys"):
        load_addm_experiment(path)


def test_save_and_load_simulation_roundtrip_npz(tmp_path):
    payload = {
        "fp_times": np.array([0.1, 0.2, 0.5]),
        "num_fpt": 1000,
        "dt": 1e-5,
        "label": "example2",
        "success": True,
    }
    path = tmp_path / "simulation.npz"
    save_simulation(path, payload)

    with np.load(path, allow_pickle=False) as archive:
        assert archive["__format__"].item() == "efpt.simulation"
        assert int(archive["__version__"].item()) == 1
        assert set(archive.files) == {
            "__format__",
            "__version__",
            "fp_times",
            "num_fpt",
            "dt",
            "label",
            "success",
        }

    loaded = load_simulation(path)
    np.testing.assert_array_equal(loaded["fp_times"], payload["fp_times"])
    assert loaded["num_fpt"] == payload["num_fpt"]
    assert loaded["dt"] == payload["dt"]
    assert loaded["label"] == payload["label"]
    assert loaded["success"] == payload["success"]


def test_simulation_io_rejects_non_npz_and_nested_or_object_payloads(tmp_path):
    with pytest.raises(ValueError, match=r"only supports '.npz' files"):
        save_simulation(tmp_path / "simulation.pkl", {"x": np.array([1.0])})

    with pytest.raises(ValueError, match=r"only supports '.npz' files"):
        load_simulation(tmp_path / "simulation.pkl")

    with pytest.raises(ValueError, match="nested dicts are not supported"):
        save_simulation(tmp_path / "bad_simulation.npz", {"nested": {"x": 1}})

    with pytest.raises(ValueError, match="unsupported object dtype"):
        save_simulation(
            tmp_path / "object_simulation.npz",
            {"bad": np.array([object()], dtype=object)},
        )
