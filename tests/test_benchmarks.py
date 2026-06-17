"""Smoke tests for the local benchmark entrypoints."""

from __future__ import annotations

import compileall
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
BENCHMARK_MODULES = [
    "benchmarks.common",
    "benchmarks.micro_logsumexp",
    "benchmarks.single_trial_backends",
    "benchmarks.single_trial_jax_methods",
    "benchmarks.batch_gpu_methods",
    "benchmarks.batch_gpu_scaling",
]
BENCHMARK_SCRIPTS = [
    BENCHMARK_DIR / "micro_logsumexp.py",
    BENCHMARK_DIR / "single_trial_backends.py",
    BENCHMARK_DIR / "single_trial_jax_methods.py",
    BENCHMARK_DIR / "batch_gpu_methods.py",
    BENCHMARK_DIR / "batch_gpu_scaling.py",
]


def _assert_metric_block(metrics):
    assert set(metrics.keys()) == {"compile", "runtime", "memory"}
    assert set(metrics["compile"].keys()) == {"forward_s", "value_and_grad_s", "total_s"}
    assert set(metrics["runtime"].keys()) == {
        "forward_s",
        "value_and_grad_s",
        "backward_proxy_s",
        "units_per_s",
        "grad_units_per_s",
    }
    assert set(metrics["memory"].keys()) == {"forward", "value_and_grad"}


def _assert_record_schema(record):
    assert {"variant", "api_kind", "workload", "compute", "metrics"} <= set(record.keys())
    _assert_metric_block(record["metrics"])


def test_benchmark_python_files_compile():
    assert compileall.compile_dir(
        str(BENCHMARK_DIR), quiet=1, force=True
    ), "benchmark folder should py_compile cleanly"


@pytest.mark.parametrize("module_name", BENCHMARK_MODULES)
def test_benchmark_modules_import_without_side_effect_output(module_name, capsys):
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_common_metric_helpers_have_stable_shape():
    common = importlib.import_module("benchmarks.common")
    metrics = common.make_metrics(runtime_forward_s=0.1)
    _assert_metric_block(metrics)
    assert common.units_per_second(10, 0.5) == pytest.approx(20.0)
    assert common.units_per_second(10, None) is None


@pytest.mark.parametrize("script_path", BENCHMARK_SCRIPTS)
@pytest.mark.benchmark_smoke
def test_benchmark_scripts_smoke_run(script_path, tmp_path):
    output_json = tmp_path / f"{script_path.stem}.json"
    completed = subprocess.run(
        [sys.executable, str(script_path), "--smoke", "--output-json", str(output_json)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"{script_path.name} smoke run failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output_json.exists(), f"{script_path.name} should emit JSON output"

    payload = json.loads(output_json.read_text())
    assert {
        "benchmark_schema_version",
        "benchmark",
        "created_at",
        "system",
        "arguments",
        "records",
    } <= set(payload.keys())
    assert payload["benchmark_schema_version"] == 2
    assert isinstance(payload["records"], list)
    for record in payload["records"]:
        _assert_record_schema(record)
