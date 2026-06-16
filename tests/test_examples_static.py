"""Static and lightweight executable regression checks for example notebooks."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
TUTORIAL_NOTEBOOK = EXAMPLES_DIR / "example_jax" / "tutorial_numpy_vs_jax.ipynb"
MLE_NOTEBOOK = EXAMPLES_DIR / "example_jax" / "mle_comparison.ipynb"
ADDM_INFERENCE_NOTEBOOK = EXAMPLES_DIR / "example4_new" / "addm_inference.ipynb"


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _notebook_text(path: Path) -> str:
    nb = _load_notebook(path)
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def _code_source(nb: dict, index: int) -> str:
    cell = nb["cells"][index]
    assert cell.get("cell_type") == "code"
    return "".join(cell.get("source", []))


@contextlib.contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _exec_notebook_cells(path: Path, cell_indices: list[int], *, prelude: str = ""):
    nb = _load_notebook(path)
    namespace = {"__name__": "__notebook__"}
    os.environ.setdefault("MPLBACKEND", "Agg")

    with _cwd(path.parent):
        exec("import matplotlib; matplotlib.use('Agg')", namespace)
        if prelude:
            exec(prelude, namespace)
        for index in cell_indices:
            exec(_code_source(nb, index), namespace)
    return namespace


def test_notebooks_do_not_reference_removed_or_private_helpers():
    banned = [
        "get_multistage_densities",
        "get_addm_fptd_jax",
        "_compute_addm_likelihoods_batchscan_core",
        "_compute_addm_loglikelihoods_batchscan_core",
        "_build_addm_mu_array_data",
        "_generate_sacc_array_data",
        "_simulate_fpt_python",
        "bench_logspace.py",
        "bench_jax_batch_gpu.py",
    ]
    for path in EXAMPLES_DIR.rglob("*.ipynb"):
        text = _notebook_text(path)
        for phrase in banned:
            assert phrase not in text, f"{path.name} still references '{phrase}'"


def test_notebooks_do_not_use_pickle_persistence():
    checked = [
        EXAMPLES_DIR / "example2" / "example2_multi-stage-approx.ipynb",
        EXAMPLES_DIR / "example3" / "example3_ou.ipynb",
        EXAMPLES_DIR / "example4_new" / "addm_gen_data.ipynb",
        EXAMPLES_DIR / "example4_new" / "addm_inference.ipynb",
        EXAMPLES_DIR / "example_jax" / "pymc_sampler_comparison.ipynb",
    ]
    for path in checked:
        if not path.exists():
            continue
        text = _notebook_text(path)
        assert ".pkl" not in text
        assert "pickle.dump" not in text
        assert "pickle.load" not in text


def test_example_notebooks_reference_current_npz_artifacts():
    addm_inference_text = _notebook_text(ADDM_INFERENCE_NOTEBOOK)
    assert "addm_data_20251015-163921.npz" in addm_inference_text
    assert "mcmc_results_20251017-035721.npz" in addm_inference_text

    for path, expected_name in [
        (EXAMPLES_DIR / "example2" / "example2_multi-stage-approx.ipynb", "ex2_fpt_data_20250305-184050.npz"),
        (EXAMPLES_DIR / "example3" / "example3_ou.ipynb", "ex3_fpt_data_20250306-155720.npz"),
        (EXAMPLES_DIR / "example4_new" / "addm_gen_data.ipynb", ".npz"),
        (EXAMPLES_DIR / "example_jax" / "pymc_sampler_comparison.ipynb", "addm_data_20251015-163921.npz"),
    ]:
        if not path.exists():
            continue
        assert expected_name in _notebook_text(path)


def test_tutorial_notebook_points_to_dedicated_benchmark_scripts():
    text = _notebook_text(TUTORIAL_NOTEBOOK)
    assert "benchmarks/single_trial_backends.py" in text
    assert "benchmarks/batch_gpu_methods.py" in text


def test_inference_comparison_notebook_uses_current_jax_wording():
    if not MLE_NOTEBOOK.exists():
        pytest.skip("mle_comparison notebook is not present in this workspace")
    text = _notebook_text(MLE_NOTEBOOK)
    assert "XLA threshold" not in text
    assert "trunc_num=6` threshold" not in text


def test_tutorial_notebook_core_cells_execute():
    ns = _exec_notebook_cells(
        TUTORIAL_NOTEBOOK,
        [2, 4, 8, 9, 20, 22],
        prelude="TRUNC_NUM_TIMING = 6\n",
    )
    assert "logfptd_multi_jax_upper" in ns
    assert "grad_sigma_multi_val" in ns


def test_addm_inference_notebook_load_cells_execute():
    ns = _exec_notebook_cells(ADDM_INFERENCE_NOTEBOOK, [3, 5])
    assert ns["num_data"] > 0
    assert ns["rt_data"].shape[0] == ns["choice_data"].shape[0]


def test_mle_comparison_notebook_setup_cells_execute():
    if not MLE_NOTEBOOK.exists():
        pytest.skip("mle_comparison notebook is not present in this workspace")
    prelude = """
NUM_TRIALS = 20
RANDOM_SEED = 42
TRUE_PARAMS = {
    "eta": 0.7,
    "kappa": 0.5,
    "a": 2.1,
    "b": 0.3,
    "x0": -0.2,
    "sigma": 1.0,
}
FIXATION_SHAPE = 6
FIXATION_SCALE = 0.1
RUN_CYTHON = False
RUN_JAX_NOGRAD = True
RUN_JAX_GRAD = True
JAX_USE_REMAT = False
TRUNC_NUM = 5
"""
    ns = _exec_notebook_cells(MLE_NOTEBOOK, [2, 4, 5, 6, 12, 13, 14, 15, 16], prelude=prelude)
    assert ns["num_data"] == 20
