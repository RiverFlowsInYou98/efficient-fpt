"""Shared helpers for the local benchmark suite.

The benchmark scripts in this folder are intentionally plain Python entrypoints
rather than a third-party benchmark framework. This module centralizes the
small amount of infrastructure they share:

- common CLI arguments
- structured JSON output
- JAX import / timing helpers
- synthetic dataset construction
- schema-stable metric shaping
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from efpt._defaults import (
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
)
from efpt.utils import resolve_quadrature_orders


BENCHMARK_SCHEMA_VERSION = 2
_MEMORY_KEYS = (
    "temp_kib",
    "total_kib",
    "argument_kib",
    "output_kib",
    "alias_kib",
)


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common benchmark CLI arguments."""
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write structured benchmark results to this JSON file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny benchmark configuration suitable for import/smoke checks.",
    )


def add_jax_precision_argument(
    parser: argparse.ArgumentParser,
    *,
    allow_both: bool = False,
    default: str = "float64",
) -> None:
    """Add a common JAX precision selector to a benchmark CLI."""
    choices = ("float32", "float64", "both") if allow_both else ("float32", "float64")
    parser.add_argument(
        "--precision",
        choices=choices,
        default=default,
        help="JAX precision mode used for this benchmark.",
    )


def add_quadrature_order_arguments(parser: argparse.ArgumentParser) -> None:
    """Add split quadrature-order CLI arguments plus the legacy alias."""
    parser.add_argument(
        "--order-mid",
        type=int,
        default=DEFAULT_MID_QUAD_ORDER,
        help="Intermediate-stage quadrature order used for q_single propagation.",
    )
    parser.add_argument(
        "--order-last",
        type=int,
        default=DEFAULT_LAST_QUAD_ORDER,
        help="Final-stage quadrature order used for fptd_single reduction.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Legacy compatibility alias that maps to both split orders.",
    )


def resolve_cli_quadrature_orders(args) -> tuple[int, int]:
    """Resolve split quadrature orders from parsed benchmark CLI args."""
    return resolve_quadrature_orders(
        order_mid=args.order_mid,
        order_last=args.order_last,
        order=getattr(args, "order", None),
    )


def resolve_precision_values(precision: str, *, smoke: bool) -> tuple[str, ...]:
    """Return the precision values a JAX benchmark should actually run."""
    if precision == "both":
        return ("float64",) if smoke else ("float32", "float64")
    return (precision,)


def import_jax(precision: str = "float64"):
    """Import JAX lazily and configure the requested precision."""
    import jax

    from efpt.jax.utils import set_jax_precision

    set_jax_precision(precision == "float64")

    import jax.numpy as jnp

    return jax, jnp


def block_tree(jax, value):
    """Recursively block on any JAX arrays inside a return value."""
    return jax.tree_util.tree_map(jax.block_until_ready, value)


def best_time(
    fn,
    *,
    args=(),
    kwargs=None,
    repeat: int = 3,
    n_calls: int = 10,
    block=None,
):
    """Return the best average wall time over repeated timing loops."""
    kwargs = {} if kwargs is None else kwargs
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            out = fn(*args, **kwargs)
            if block is not None:
                block(out)
        elapsed = (time.perf_counter() - t0) / n_calls
        best = min(best, elapsed)
    return best


def xla_memory_kib(compiled) -> dict[str, float | None]:
    """Extract a small set of XLA memory-analysis metrics in KiB."""
    try:
        mem = compiled.memory_analysis()
        if mem is None:
            return {key: None for key in _MEMORY_KEYS}
        arg_bytes = getattr(mem, "argument_size_in_bytes", 0) or 0
        out_bytes = getattr(mem, "output_size_in_bytes", 0) or 0
        temp_bytes = getattr(mem, "temp_size_in_bytes", 0) or 0
        alias_bytes = getattr(mem, "alias_size_in_bytes", 0) or 0
        return {
            "temp_kib": temp_bytes / 1024.0,
            "total_kib": (arg_bytes + out_bytes + temp_bytes - alias_bytes) / 1024.0,
            "argument_kib": arg_bytes / 1024.0,
            "output_kib": out_bytes / 1024.0,
            "alias_kib": alias_bytes / 1024.0,
        }
    except Exception:
        return {key: None for key in _MEMORY_KEYS}


def units_per_second(units: float | int | None, runtime_s: float | None) -> float | None:
    """Compute throughput in units/s when both inputs are available."""
    if units is None or runtime_s is None or runtime_s <= 0.0:
        return None
    return float(units) / float(runtime_s)


def summarize_value(value):
    """Return a compact JSON-safe summary of a benchmark output."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def summarize_gradient_tree(gradients):
    """Return a compact summary of a small gradient tuple/tree."""
    leaves = []

    def _collect(node):
        if isinstance(node, dict):
            for value in node.values():
                _collect(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _collect(value)
        else:
            leaves.append(np.asarray(node, dtype=np.float64).ravel())

    _collect(gradients)
    if not leaves:
        return {"l2_norm": 0.0, "max_abs": 0.0, "values": []}

    flat = np.concatenate(leaves)
    values = flat.tolist() if flat.size <= 8 else None
    return {
        "l2_norm": float(np.linalg.norm(flat)),
        "max_abs": float(np.max(np.abs(flat))),
        "values": values,
    }


def make_metrics(
    *,
    compile_forward_s: float | None = None,
    compile_value_and_grad_s: float | None = None,
    runtime_forward_s: float | None = None,
    runtime_value_and_grad_s: float | None = None,
    backward_proxy_s: float | None = None,
    units_per_s_value: float | None = None,
    grad_units_per_s_value: float | None = None,
    forward_memory: dict[str, float | None] | None = None,
    value_and_grad_memory: dict[str, float | None] | None = None,
) -> dict[str, object]:
    """Build one stable benchmark metrics block."""
    return {
        "compile": {
            "forward_s": compile_forward_s,
            "value_and_grad_s": compile_value_and_grad_s,
            "total_s": (
                None
                if compile_forward_s is None and compile_value_and_grad_s is None
                else (compile_forward_s or 0.0) + (compile_value_and_grad_s or 0.0)
            ),
        },
        "runtime": {
            "forward_s": runtime_forward_s,
            "value_and_grad_s": runtime_value_and_grad_s,
            "backward_proxy_s": backward_proxy_s,
            "units_per_s": units_per_s_value,
            "grad_units_per_s": grad_units_per_s_value,
        },
        "memory": {
            "forward": forward_memory,
            "value_and_grad": value_and_grad_memory,
        },
    }


def benchmark_jax_forward_callable(
    jax,
    fn,
    params,
    *,
    n_calls: int = 10,
    repeat: int = 3,
):
    """Compile and time the forward path of a JAX callable."""
    fwd_callable = fn if hasattr(fn, "lower") else jax.jit(fn)

    t0 = time.perf_counter()
    fwd_compiled = fwd_callable.lower(*params).compile()
    fwd_compile_s = time.perf_counter() - t0

    forward_value = fwd_compiled(*params)
    block_tree(jax, forward_value)
    forward_s = best_time(
        fwd_compiled,
        args=params,
        n_calls=n_calls,
        repeat=repeat,
        block=lambda out: block_tree(jax, out),
    )

    return {
        "compile": {"forward_s": fwd_compile_s},
        "runtime": {"forward_s": forward_s},
        "memory": {"forward": xla_memory_kib(fwd_compiled)},
        "forward_value": forward_value,
    }


def benchmark_jax_scalar_objective(
    jax,
    fn,
    params,
    *,
    n_calls: int = 10,
    repeat: int = 3,
    grad_argnums,
):
    """Compile and time a scalar-output JAX objective and its gradients."""
    forward = benchmark_jax_forward_callable(
        jax,
        fn,
        params,
        n_calls=n_calls,
        repeat=repeat,
    )

    vg_callable = jax.jit(jax.value_and_grad(fn, argnums=grad_argnums))
    t0 = time.perf_counter()
    vg_compiled = vg_callable.lower(*params).compile()
    vg_compile_s = time.perf_counter() - t0

    value_and_grad = vg_compiled(*params)
    block_tree(jax, value_and_grad)
    value_and_grad_s = best_time(
        vg_compiled,
        args=params,
        n_calls=n_calls,
        repeat=repeat,
        block=lambda out: block_tree(jax, out),
    )

    return {
        "compile": {
            "forward_s": forward["compile"]["forward_s"],
            "value_and_grad_s": vg_compile_s,
        },
        "runtime": {
            "forward_s": forward["runtime"]["forward_s"],
            "value_and_grad_s": value_and_grad_s,
            "backward_proxy_s": max(
                value_and_grad_s - forward["runtime"]["forward_s"], 0.0
            ),
        },
        "memory": {
            "forward": forward["memory"]["forward"],
            "value_and_grad": xla_memory_kib(vg_compiled),
        },
        "forward_value": forward["forward_value"],
        "value_and_grad": value_and_grad,
    }


def format_kib(value: float | None) -> str:
    """Format a memory value in KiB for console tables."""
    if value is None:
        return "N/A"
    return f"{value:8.1f}"


def format_rate(value: float | None) -> str:
    """Format a throughput value for console tables."""
    if value is None:
        return "N/A"
    return f"{value:10.1f}"


def to_builtin(value):
    """Recursively convert numpy/JAX scalars and arrays to JSON-safe builtins."""
    if isinstance(value, dict):
        return {key: to_builtin(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def runtime_info(jax_module=None) -> dict[str, object]:
    """Collect a small amount of runtime metadata for JSON output."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if jax_module is not None:
        info["jax_devices"] = [
            {
                "platform": device.platform,
                "device_kind": getattr(device, "device_kind", str(device)),
            }
            for device in jax_module.devices()
        ]
    return info


def write_json_output(
    path: Path | None,
    *,
    benchmark: str,
    records,
    args,
    jax_module=None,
    extra=None,
):
    """Write one benchmark run as a structured JSON document."""
    if path is None:
        return

    payload = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": benchmark,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": runtime_info(jax_module),
        "arguments": to_builtin(vars(args)),
        "records": to_builtin(records),
    }
    if extra:
        payload.update(to_builtin(extra))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_single_trial_test_data_np(d: int):
    """Build a deterministic single-trial test case for Cython benchmarks."""
    sigma, a, b, x0 = 1.0, 1.5, 0.3, 0.0
    mu = np.array([0.1 * ((-1) ** i) for i in range(d)], dtype=np.float64)
    node = np.array([0.3 * i for i in range(d)], dtype=np.float64)
    t = float(node[-1] + 0.5)
    sigma_arr = np.full(d, sigma, dtype=np.float64)
    b1_arr = np.full(d, -b, dtype=np.float64)
    b2_arr = np.full(d, b, dtype=np.float64)
    return t, mu, node, sigma, a, b, x0, sigma_arr, b1_arr, b2_arr


def make_single_trial_test_data_jax(jnp, d: int, max_d: int | None = None):
    """Build a deterministic single-trial test case for JAX benchmarks."""
    from efpt.jax.utils import get_jax_dtype

    sigma, a, b, x0 = 1.0, 1.5, 0.3, 0.0
    if max_d is None:
        max_d = max(d + 2, 10)
    dtype = get_jax_dtype()

    mu_list = [0.1 * ((-1) ** i) for i in range(d)]
    node_list = [0.3 * i for i in range(d)]
    t = float(node_list[-1] + 0.5)

    mu_arr = jnp.array(np.pad(mu_list, (0, max_d - d)), dtype=dtype)
    node_arr = jnp.array(np.pad(node_list, (0, max_d - d)), dtype=dtype)
    sigma_arr = jnp.array(
        np.pad([sigma] * d, (0, max_d - d), constant_values=sigma),
        dtype=dtype,
    )
    b1_arr = jnp.array(
        np.pad([-b] * d, (0, max_d - d), constant_values=-b),
        dtype=dtype,
    )
    b2_arr = jnp.array(
        np.pad([b] * d, (0, max_d - d), constant_values=b),
        dtype=dtype,
    )
    return {
        "t": jnp.asarray(t, dtype=dtype),
        "d": int(d),
        "mu_arr": mu_arr,
        "node_arr": node_arr,
        "sigma": jnp.asarray(sigma, dtype=dtype),
        "a": jnp.asarray(a, dtype=dtype),
        "b": jnp.asarray(b, dtype=dtype),
        "x0": jnp.asarray(x0, dtype=dtype),
        "sigma_arr": sigma_arr,
        "b1_arr": b1_arr,
        "b2_arr": b2_arr,
        "r1": jnp.asarray(0.1, dtype=dtype),
        "r2": jnp.asarray(0.1, dtype=dtype),
        "flag": jnp.int32(0),
        "max_d": int(max_d),
    }


def make_batch_addm_dataset(jnp, n_trials: int, max_d: int, seed: int = 0):
    """Build a deterministic padded batch of aDDM trials for JAX benchmarks."""
    rng = np.random.default_rng(seed)
    d_data = rng.integers(1, max_d + 1, size=n_trials, dtype=np.int32)
    choice_data = rng.choice(np.array([1, -1], dtype=np.int32), size=n_trials)
    r1_data = rng.uniform(0.0, 1.0, size=n_trials).astype(np.float64)
    r2_data = rng.uniform(0.0, 1.0, size=n_trials).astype(np.float64)
    flag_data = rng.integers(0, 2, size=n_trials, dtype=np.int32)
    sacc_array_data = np.zeros((n_trials, max_d), dtype=np.float64)
    rt_data = np.zeros(n_trials, dtype=np.float64)

    for idx, d in enumerate(d_data):
        if d > 1:
            dt = rng.uniform(0.2, 0.7, size=d - 1)
            sacc_array_data[idx, 1:d] = np.cumsum(dt)
        rt_data[idx] = sacc_array_data[idx, d - 1] + rng.uniform(0.2, 0.8)

    return {
        "rt_data": jnp.array(rt_data),
        "choice_data": jnp.array(choice_data),
        "r1_data": jnp.array(r1_data),
        "r2_data": jnp.array(r2_data),
        "flag_data": jnp.array(flag_data),
        "sacc_array_data": jnp.array(sacc_array_data),
        "d_data": jnp.array(d_data),
        "n_trials": int(n_trials),
        "max_d": int(max_d),
        "seed": int(seed),
    }
