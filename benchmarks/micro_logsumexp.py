"""Micro-benchmark for the NumPy multistage logsumexp helper."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from benchmarks.common import (
    add_common_cli_arguments,
    best_time,
    make_metrics,
    write_json_output,
)
from efficient_fpt.numpy.multi_stage import _logsumexp as numpy_logsumexp

try:
    from scipy.special import logsumexp as scipy_logsumexp
except ImportError:  # pragma: no cover - depends on optional extra
    scipy_logsumexp = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to build deterministic benchmark arrays.",
    )
    return parser.parse_args()


def benchmark_cases(smoke: bool):
    if smoke:
        return [
            ((30,), None, 200),
            ((32, 16), 1, 100),
        ]
    return [
        ((30,), None, 20000),
        ((30, 30), 1, 10000),
        ((64, 30), 1, 5000),
        ((256, 30), 1, 2000),
    ]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    records = []

    print("=" * 72)
    print("NUMPY LOGSUMEXP MICRO-BENCHMARK")
    print("=" * 72)

    if scipy_logsumexp is None:
        print("SciPy is not installed; skipping comparison.")
        write_json_output(
            args.output_json,
            benchmark="micro_logsumexp",
            records=[],
            args=args,
            extra={"status": "skipped", "reason": "scipy_not_installed"},
        )
        return

    print(
        f"{'shape':18s}  {'axis':>4s}  {'variant':18s}  {'time (us)':>10s}  {'ratio vs scipy':>14s}"
    )
    for shape, axis, n_calls in benchmark_cases(args.smoke):
        arr = rng.normal(size=shape)
        arr.reshape(-1)[::7] = -np.inf

        scipy_time = best_time(
            scipy_logsumexp,
            args=(arr,),
            kwargs={"axis": axis},
            repeat=5 if not args.smoke else 2,
            n_calls=n_calls,
        )
        numpy_time = best_time(
            numpy_logsumexp,
            args=(arr,),
            kwargs={"axis": axis},
            repeat=5 if not args.smoke else 2,
            n_calls=n_calls,
        )

        records.extend(
            [
                {
                    "variant": "scipy_logsumexp",
                    "api_kind": "micro_kernel",
                    "workload": {"shape": list(shape), "axis": axis},
                    "compute": {"n_calls": n_calls},
                    "metrics": {
                        **make_metrics(runtime_forward_s=scipy_time),
                    },
                    "values": {"ratio_vs_scipy": 1.0},
                },
                {
                    "variant": "numpy_logsumexp",
                    "api_kind": "micro_kernel",
                    "workload": {"shape": list(shape), "axis": axis},
                    "compute": {"n_calls": n_calls},
                    "metrics": {
                        **make_metrics(runtime_forward_s=numpy_time),
                    },
                    "values": {"ratio_vs_scipy": numpy_time / scipy_time},
                },
            ]
        )

        print(
            f"{str(shape):18s}  {str(axis):>4s}  "
            f"{'scipy_logsumexp':18s}  {scipy_time * 1e6:10.1f}  {1.0:14.3f}"
        )
        print(
            f"{'':18s}  {'':>4s}  "
            f"{'numpy_logsumexp':18s}  {numpy_time * 1e6:10.1f}  {numpy_time / scipy_time:14.3f}"
        )

    write_json_output(
        args.output_json,
        benchmark="micro_logsumexp",
        records=records,
        args=args,
    )


if __name__ == "__main__":
    main()
