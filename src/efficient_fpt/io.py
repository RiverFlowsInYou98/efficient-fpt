"""Save / load helpers for simulation and grouped aDDM experiment data."""

import pickle

import numpy as np


# ---------------------------------------------------------------------------
# Generic simulation save / load (any dict)
# ---------------------------------------------------------------------------


def save_simulation(path, data):
    """Save a simulation data dict to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_simulation(path):
    """Load a simulation data dict from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


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


def _validate_array_group(group_name, group, required_keys, *, expected_len=None):
    """Validate a grouped array payload and return its leading length."""
    missing = sorted(required_keys - set(group.keys()))
    if missing:
        raise ValueError(f"Missing required keys in '{group_name}': {missing}")

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


def _validate_addm_experiment(data):
    """Check that an aDDM experiment dict is fully canonical and shape-consistent."""
    missing_groups = sorted(_GROUP_KEYS - set(data.keys()))
    if missing_groups:
        raise ValueError(f"Missing required groups: {missing_groups}")

    for group_name in _GROUP_KEYS:
        if not isinstance(data[group_name], dict):
            raise ValueError(f"'{group_name}' must be a dict")

    n_trials = _validate_array_group(
        "decision_data", data["decision_data"], _DECISION_KEYS
    )
    _validate_array_group(
        "covariates", data["covariates"], _COVARIATE_KEYS, expected_len=n_trials
    )

    missing_params = sorted(_PARAM_KEYS - set(data["params"].keys()))
    if missing_params:
        raise ValueError(f"Missing required keys in 'params': {missing_params}")

    missing_config = sorted(_CONFIG_KEYS - set(data["config"].keys()))
    if missing_config:
        raise ValueError(f"Missing required keys in 'config': {missing_config}")


def _canonicalize_addm_experiment(data):
    """Return a validated grouped aDDM experiment dict while preserving extras."""
    canonical = dict(data)
    for group_name in _GROUP_KEYS:
        if group_name in canonical:
            if not isinstance(canonical[group_name], dict):
                raise ValueError(f"'{group_name}' must be a dict")
            canonical[group_name] = dict(canonical[group_name])
    _validate_addm_experiment(canonical)
    return canonical


def save_addm_experiment(path, data):
    """Validate and save a grouped canonical aDDM experiment dict."""
    canonical = _canonicalize_addm_experiment(data)
    with open(path, "wb") as f:
        pickle.dump(canonical, f)


def load_addm_experiment(path):
    """Load a grouped canonical aDDM experiment dict."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return _canonicalize_addm_experiment(raw)
