"""Backward-compatible re-export of the public NumPy multi-stage API."""

from .numpy.multi_stage import (
    compute_homog_multistage_logfptds_and_lognpd,
)

__all__ = [
    "compute_homog_multistage_logfptds_and_lognpd",
]
