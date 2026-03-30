"""Static regression checks for example notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
TUTORIAL_NOTEBOOK = EXAMPLES_DIR / "example_jax" / "tutorial_numpy_vs_jax.ipynb"


def _notebook_text(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def test_notebooks_do_not_reference_removed_or_private_helpers():
    banned = [
        "get_multistage_densities",
        "get_addm_fptd_jax",
        "_compute_addm_likelihoods_batchscan_core",
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


def test_tutorial_notebook_points_to_dedicated_benchmark_scripts():
    text = _notebook_text(TUTORIAL_NOTEBOOK)
    assert "benchmarks/single_trial_backends.py" in text
    assert "benchmarks/batch_gpu_methods.py" in text


def test_inference_comparison_notebook_uses_current_jax_wording():
    text = _notebook_text(EXAMPLES_DIR / "inference_comparison.ipynb")
    assert "XLA threshold" not in text
    assert "trunc_num=6` threshold" not in text
