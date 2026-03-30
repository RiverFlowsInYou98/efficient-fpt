"""Shared test helper functions (non-fixture)."""

import numpy as np


def try_import_cython_multi_stage():
    """Try importing Cython multi_stage functions.

    Returns (compute_addm_fptd, compute_heterog_multistage_fptd) or (None, None).
    """
    try:
        from efficient_fpt.cython.multi_stage import (
            compute_addm_fptd,
            compute_heterog_multistage_fptd,
        )
        return compute_addm_fptd, compute_heterog_multistage_fptd
    except ImportError:
        return None, None


def mu_to_addm_covariates(mu1_data, mu2_data):
    """Convert pre-derived mu1/mu2 arrays to public ADDM covariates.

    Uses eta=0, kappa=1 so that mu1 = r1 and mu2 = -r2,
    giving the caller direct control over drifts.

    Returns (eta, kappa, r1_data, r2_data, flag_data).
    """
    r1_data = np.asarray(mu1_data, dtype=np.float64)
    r2_data = -np.asarray(mu2_data, dtype=np.float64)
    flag_data = np.zeros(len(r1_data), dtype=np.int32)
    return 0.0, 1.0, r1_data, r2_data, flag_data
