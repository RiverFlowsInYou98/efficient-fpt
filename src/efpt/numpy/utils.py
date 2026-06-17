"""NumPy utility helpers for efficient-fpt."""

from __future__ import annotations

import numpy as np


def positive_log(values):
    """Return log(x) for x > 0, -inf for x <= 0, and nan when x is nan."""
    values = np.asarray(values, dtype=np.float64)
    result = np.where(values > 0.0, np.log(np.maximum(values, 1e-300)), -np.inf)
    result = np.where(np.isnan(values), np.nan, result)
    return float(result) if result.ndim == 0 else result
