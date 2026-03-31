"""Safe `.npz` save / load helpers for simulation and grouped aDDM data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Archive metadata
# ---------------------------------------------------------------------------


_ARCHIVE_VERSION = 1
_SIMULATION_FORMAT = "efficient_fpt.simulation"
_ADDM_EXPERIMENT_FORMAT = "efficient_fpt.addm_experiment"
_RESERVED_ARCHIVE_KEYS = {"__format__", "__version__"}


# ---------------------------------------------------------------------------
# Generic simulation save / load (flat dict only)
# ---------------------------------------------------------------------------


def _require_npz_path(path, *, what):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"{what} only supports '.npz' files, got '{path.name}'")
    return path


def _load_archive_metadata(archive, *, expected_format):
    files = set(archive.files)
    missing = sorted(_RESERVED_ARCHIVE_KEYS - files)
    if missing:
        raise ValueError(f"Archive is missing required metadata keys: {missing}")

    archive_format = archive["__format__"].item()
    if archive_format != expected_format:
        raise ValueError(
            f"Archive format mismatch: expected '{expected_format}', got '{archive_format}'"
        )

    version = int(np.asarray(archive["__version__"]).item())
    if version != _ARCHIVE_VERSION:
        raise ValueError(
            f"Archive version mismatch: expected {_ARCHIVE_VERSION}, got {version}"
        )


def _validate_simulation_value(key, value):
    if isinstance(value, dict):
        raise ValueError(
            f"Simulation value '{key}' must be flat; nested dicts are not supported"
        )
    arr = np.asarray(value)
    if arr.dtype.kind == "O":
        raise ValueError(
            f"Simulation value '{key}' has unsupported object dtype; "
            "use numeric/bool/string scalars or non-object arrays"
        )
    return arr


def _scalarize_loaded_value(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return np.array(arr, copy=True)


def save_simulation(path, data):
    """Save a flat simulation data dict to a compressed `.npz` archive."""
    path = _require_npz_path(path, what="save_simulation")
    if not isinstance(data, dict):
        raise ValueError("Simulation payload must be a dict")

    archive = {
        "__format__": np.asarray(_SIMULATION_FORMAT),
        "__version__": np.asarray(_ARCHIVE_VERSION, dtype=np.int64),
    }
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("Simulation payload keys must be strings")
        if key in _RESERVED_ARCHIVE_KEYS:
            raise ValueError(f"Simulation payload key '{key}' is reserved")
        if "/" in key:
            raise ValueError(
                f"Simulation payload key '{key}' is not flat; '/' is not allowed"
            )
        archive[key] = _validate_simulation_value(key, value)

    np.savez_compressed(path, **archive)


def load_simulation(path):
    """Load a flat simulation data dict from a compressed `.npz` archive."""
    path = _require_npz_path(path, what="load_simulation")
    with np.load(path, allow_pickle=False) as archive:
        _load_archive_metadata(archive, expected_format=_SIMULATION_FORMAT)
        invalid = sorted(
            key
            for key in archive.files
            if key not in _RESERVED_ARCHIVE_KEYS and "/" in key
        )
        if invalid:
            raise ValueError(
                f"Simulation archive must be flat; found nested keys: {invalid}"
            )
        return {
            key: _scalarize_loaded_value(archive[key])
            for key in archive.files
            if key not in _RESERVED_ARCHIVE_KEYS
        }


# ---------------------------------------------------------------------------
# aDDM experiment data: grouped canonical format
# ---------------------------------------------------------------------------


_GROUP_KEYS = {"decision_data", "params", "covariates", "config"}
_DECISION_KEYS = {"rt_data", "choice_data"}
_PARAM_KEYS = {"eta", "kappa", "sigma", "a", "b", "x0"}
_COVARIATE_KEYS = {
    "r1_data",
    "r2_data",
    "flag_data",
    "sacc_array_data",
    "d_data",
}
_CONFIG_KEYS = {"dt", "T", "gamma_shape", "gamma_scale", "r_range"}


def _validate_key_set(container_name, actual_keys, required_keys):
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise ValueError(f"Missing required keys in '{container_name}': {missing}")

    extra = sorted(actual_keys - required_keys)
    if extra:
        raise ValueError(f"Unexpected keys in '{container_name}': {extra}")


def _validate_array_group(group_name, group, required_keys, *, expected_len=None):
    """Validate a grouped array payload and return its leading length."""
    _validate_key_set(group_name, set(group.keys()), required_keys)

    group_len = expected_len
    for key in required_keys:
        arr = np.asarray(group[key])
        if arr.ndim == 0:
            raise ValueError(f"Array '{group_name}.{key}' must be at least 1-dimensional")
        if group_len is None:
            group_len = arr.shape[0]
        elif arr.shape[0] != group_len:
            raise ValueError(
                f"Array '{group_name}.{key}' has length {arr.shape[0]}, expected {group_len}"
            )
    return group_len


def _validate_scalar(value, *, name):
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"'{name}' must be scalar-valued")
    if arr.dtype.kind == "O":
        raise ValueError(f"'{name}' has unsupported object dtype")


def _validate_r_range(value):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape[0] != 2:
        raise ValueError("'config.r_range' must contain exactly two values")
    if arr.dtype.kind == "O":
        raise ValueError("'config.r_range' has unsupported object dtype")


def _validate_addm_experiment(data):
    """Check that an aDDM experiment dict is canonical and shape-consistent."""
    if not isinstance(data, dict):
        raise ValueError("aDDM experiment payload must be a dict")

    _validate_key_set("payload", set(data.keys()), _GROUP_KEYS)

    for group_name in _GROUP_KEYS:
        if not isinstance(data[group_name], dict):
            raise ValueError(f"'{group_name}' must be a dict")

    n_trials = _validate_array_group(
        "decision_data", data["decision_data"], _DECISION_KEYS
    )
    _validate_array_group(
        "covariates", data["covariates"], _COVARIATE_KEYS, expected_len=n_trials
    )

    _validate_key_set("params", set(data["params"].keys()), _PARAM_KEYS)
    for key in _PARAM_KEYS:
        _validate_scalar(data["params"][key], name=f"params.{key}")

    _validate_key_set("config", set(data["config"].keys()), _CONFIG_KEYS)
    for key in _CONFIG_KEYS - {"r_range"}:
        _validate_scalar(data["config"][key], name=f"config.{key}")
    _validate_r_range(data["config"]["r_range"])


def _canonicalize_addm_experiment(data):
    """Return a validated grouped canonical aDDM experiment dict."""
    _validate_addm_experiment(data)
    canonical = {}
    for group_name in _GROUP_KEYS:
        group = data[group_name]
        canonical[group_name] = dict(group)
    return canonical


def _flatten_addm_experiment(data):
    canonical = _canonicalize_addm_experiment(data)
    archive = {
        "__format__": np.asarray(_ADDM_EXPERIMENT_FORMAT),
        "__version__": np.asarray(_ARCHIVE_VERSION, dtype=np.int64),
    }

    for key in _DECISION_KEYS:
        archive[f"decision_data/{key}"] = np.asarray(canonical["decision_data"][key])
    for key in _PARAM_KEYS:
        archive[f"params/{key}"] = np.asarray(canonical["params"][key])
    for key in _COVARIATE_KEYS:
        archive[f"covariates/{key}"] = np.asarray(canonical["covariates"][key])
    for key in _CONFIG_KEYS - {"r_range"}:
        archive[f"config/{key}"] = np.asarray(canonical["config"][key])
    archive["config/r_range"] = np.asarray(tuple(canonical["config"]["r_range"]))
    return archive


def _expected_addm_archive_keys():
    expected = set(_RESERVED_ARCHIVE_KEYS)
    expected.update(f"decision_data/{key}" for key in _DECISION_KEYS)
    expected.update(f"params/{key}" for key in _PARAM_KEYS)
    expected.update(f"covariates/{key}" for key in _COVARIATE_KEYS)
    expected.update(f"config/{key}" for key in _CONFIG_KEYS)
    return expected


def _load_addm_archive(archive):
    files = set(archive.files)
    expected = _expected_addm_archive_keys()

    missing = sorted(expected - files)
    if missing:
        raise ValueError(f"Archive is missing required experiment keys: {missing}")

    extra = sorted(files - expected)
    if extra:
        raise ValueError(f"Archive has unexpected experiment keys: {extra}")

    data = {
        "decision_data": {},
        "params": {},
        "covariates": {},
        "config": {},
    }

    for key in _DECISION_KEYS:
        data["decision_data"][key] = np.array(archive[f"decision_data/{key}"], copy=True)
    for key in _PARAM_KEYS:
        data["params"][key] = _scalarize_loaded_value(archive[f"params/{key}"])
    for key in _COVARIATE_KEYS:
        data["covariates"][key] = np.array(archive[f"covariates/{key}"], copy=True)
    for key in _CONFIG_KEYS - {"r_range"}:
        data["config"][key] = _scalarize_loaded_value(archive[f"config/{key}"])
    data["config"]["r_range"] = tuple(np.asarray(archive["config/r_range"]).tolist())
    return _canonicalize_addm_experiment(data)


def save_addm_experiment(path, data):
    """Validate and save a grouped canonical aDDM experiment dict to `.npz`."""
    path = _require_npz_path(path, what="save_addm_experiment")
    archive = _flatten_addm_experiment(data)
    np.savez_compressed(path, **archive)


def load_addm_experiment(path):
    """Load a grouped canonical aDDM experiment dict from `.npz`."""
    path = _require_npz_path(path, what="load_addm_experiment")
    with np.load(path, allow_pickle=False) as archive:
        _load_archive_metadata(archive, expected_format=_ADDM_EXPERIMENT_FORMAT)
        return _load_addm_archive(archive)
