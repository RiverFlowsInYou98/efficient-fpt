"""Save / load helpers for simulation and aDDM experiment data."""

import pickle
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Generic simulation save / load (any dict)
# ---------------------------------------------------------------------------

def save_simulation(path, data):
    """Save a simulation data dict to a pickle file.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    data : dict
        Arbitrary data dict.
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_simulation(path):
    """Load a simulation data dict from a pickle file.

    Parameters
    ----------
    path : str or Path
        Path to a pickle file.

    Returns
    -------
    dict
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# aDDM experiment data: canonical format, validation, legacy migration
# ---------------------------------------------------------------------------

# Canonical per-trial array keys
_ARRAY_KEYS = {
    "rt", "choice",
    "mu_array_data", "sacc_array_data", "d_data",
    "r1_data", "r2_data", "flag_data",
    "mu1_data", "mu2_data",
}

# Scalar parameter keys expected inside "params"
_PARAM_KEYS = {
    "eta", "kappa", "sigma", "a", "b", "x0",
    "dt", "T", "gamma_shape", "gamma_scale", "r_range",
}

# Keys that need _data suffix migration
_SUFFIX_RENAMES = {
    "r1": "r1_data",
    "r2": "r2_data",
    "flag": "flag_data",
    "mu1": "mu1_data",
    "mu2": "mu2_data",
}

# Legacy "_padded_" array renames
_PADDED_RENAMES = {
    "mu_array_padded_data": "mu_array_data",
    "sacc_array_padded_data": "sacc_array_data",
}

# Legacy param key renames
_PARAM_RENAMES = {
    "gamma_shape_param": "gamma_shape",
    "gamma_scale_param": "gamma_scale",
}


def _migrate_addm_legacy(raw):
    """Migrate a legacy aDDM dict to canonical format.

    Handles legacy patterns:
    1. Combined ``decision_data`` → split into ``rt`` and ``choice``
    2. Flat scalar params → nest into ``params`` dict
    3. Missing ``_data`` suffix on per-trial keys
    4. ``_padded_`` array names → canonical names
    5. Missing ``mu1_data`` / ``mu2_data`` → recompute from params
    """
    data = dict(raw)  # shallow copy

    # --- Pattern 1: combined decision_data → separate rt / choice ---
    if "decision_data" in data and "rt" not in data:
        dd = data.pop("decision_data")
        data["rt"] = np.ascontiguousarray(dd[:, 0], dtype=np.float64)
        data["choice"] = np.ascontiguousarray(dd[:, 1], dtype=np.int32)

    # --- Pattern 2: flat params → nested ---
    if "params" not in data:
        params = {}
        # Collect known param keys (including legacy renames)
        for key in list(data.keys()):
            canonical = _PARAM_RENAMES.get(key, key)
            if canonical in _PARAM_KEYS and not isinstance(data[key], np.ndarray):
                params[canonical] = data.pop(key)
        if params:
            data["params"] = params

    # --- Pattern 3: missing _data suffix ---
    for old, new in _SUFFIX_RENAMES.items():
        if old in data and new not in data:
            data[new] = data.pop(old)

    # --- Pattern 4: _padded_ array names → canonical ---
    for old, new in _PADDED_RENAMES.items():
        if old in data and new not in data:
            data[new] = data.pop(old)

    # --- Pattern 5: recompute mu1_data / mu2_data if missing ---
    if "mu1_data" not in data and "params" in data:
        p = data["params"]
        if all(k in p for k in ("eta", "kappa")) and "r1_data" in data:
            r1 = np.asarray(data["r1_data"], dtype=np.float64)
            r2 = np.asarray(data["r2_data"], dtype=np.float64)
            data["mu1_data"] = p["kappa"] * (r1 - p["eta"] * r2)
            data["mu2_data"] = p["kappa"] * (p["eta"] * r1 - r2)

    return data


def _validate_addm_experiment(data):
    """Check that required aDDM keys are present and shapes are consistent."""
    missing = {"rt", "choice", "d_data", "params"} - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    n = len(data["rt"])
    for key in _ARRAY_KEYS:
        if key in data:
            arr = data[key]
            if hasattr(arr, "__len__") and len(arr) != n:
                raise ValueError(
                    f"Array '{key}' has length {len(arr)}, expected {n}"
                )


def save_addm_experiment(path, data):
    """Validate and save an aDDM experiment data dict to a pickle file.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    data : dict
        aDDM experiment data dict in canonical format.
    """
    _validate_addm_experiment(data)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_addm_experiment(path):
    """Load an aDDM experiment data dict, auto-migrating legacy formats.

    Parameters
    ----------
    path : str or Path
        Path to a pickle file.

    Returns
    -------
    dict
        aDDM experiment data in canonical format.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return _migrate_addm_legacy(raw)
