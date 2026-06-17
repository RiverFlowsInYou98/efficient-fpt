"""Scaling benchmark for the production JAX batch MCMC path."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from benchmarks.common import (
    add_common_cli_arguments,
    add_jax_precision_argument,
    add_quadrature_order_arguments,
    benchmark_jax_scalar_objective,
    format_kib,
    format_rate,
    import_jax,
    make_batch_addm_dataset,
    make_metrics,
    resolve_cli_quadrature_orders,
    resolve_precision_values,
    summarize_gradient_tree,
    summarize_value,
    units_per_second,
    write_json_output,
)


DEFAULT_PARAMS = (0.25, 1.1, 1.0, 1.5, 0.3, 0.0)
BASE_WORKLOAD = {"n_trials": 512, "max_d": 8}
BASE_COMPUTE = {
    "order_mid": 20,
    "order_last": 30,
    "trunc_num": 50,
    "log_space": False,
    "precision": "float64",
}
SWEEPS = {
    "n_trials": [128, 512, 1024],
    "max_d": [4, 8, 12],
    "order_mid": [10, 20, 30],
    "order_last": [20, 30, 40],
    "trunc_num": [20, 50, 80],
    "log_space": [False, True],
    "precision": ["float32", "float64"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(parser)
    add_jax_precision_argument(parser, allow_both=True, default="both")
    add_quadrature_order_arguments(parser)
    parser.add_argument(
        "--n-calls",
        type=int,
        default=6,
        help="Number of timing calls after compilation.",
    )
    args = parser.parse_args()
    args.order_mid, args.order_last = resolve_cli_quadrature_orders(args)
    return args

def sweep_values(smoke: bool, precision_values: tuple[str, ...]):
    if not smoke:
        sweeps = dict(SWEEPS)
        sweeps["precision"] = list(precision_values)
        return sweeps
    return {
        "n_trials": [32],
        "log_space": [False],
    }


def main():
    args = parse_args()
    n_calls = 1 if args.smoke else args.n_calls
    precision_values = resolve_precision_values(args.precision, smoke=args.smoke)
    base_workload = {"n_trials": 64, "max_d": 4} if args.smoke else BASE_WORKLOAD
    base_precision = "float64" if args.precision == "both" else precision_values[0]
    base_compute = (
        {
            "order_mid": args.order_mid,
            "order_last": args.order_last,
            "trunc_num": 20,
            "log_space": False,
            "precision": base_precision,
        }
        if args.smoke
        else {
            **BASE_COMPUTE,
            "order_mid": args.order_mid,
            "order_last": args.order_last,
            "precision": base_precision,
        }
    )
    records = []
    jax = None

    print("=" * 104)
    print("BATCH GPU SCALING BENCHMARK")
    print("=" * 104)
    print("This benchmark reports forward, value-and-grad, backward proxy, and XLA memory.")

    for sweep_name, values in sweep_values(args.smoke, precision_values).items():
        print("\n" + "-" * 104)
        print(f"sweep={sweep_name}")
        print("-" * 104)
        print(
            f"{'variant':20s}  {'value':>10s}  {'compile(s)':>10s}  "
            f"{'fwd(ms)':>10s}  {'v+g(ms)':>10s}  {'bwd_proxy(ms)':>14s}  "
            f"{'trials/s':>10s}  {'temp(KiB)':>10s}"
        )

        for value_idx, sweep_value in enumerate(values):
            workload = dict(base_workload)
            compute = dict(base_compute)

            if sweep_name in workload:
                workload[sweep_name] = sweep_value
            else:
                compute[sweep_name] = sweep_value

            jax, jnp = import_jax(compute["precision"])
            from efpt.jax.batch import make_addm_nll_function_batchscan

            dataset = make_batch_addm_dataset(
                jnp,
                n_trials=workload["n_trials"],
                max_d=workload["max_d"],
                seed=2000 + value_idx + 100 * len(records),
            )
            params = tuple(
                (jnp.float32 if compute["precision"] == "float32" else jnp.float64)(value)
                for value in DEFAULT_PARAMS
            )

            factories = {
                "batchscan_nll": make_addm_nll_function_batchscan(
                    dataset["rt_data"],
                    dataset["choice_data"],
                    dataset["r1_data"],
                    dataset["r2_data"],
                    dataset["flag_data"],
                    dataset["sacc_array_data"],
                    dataset["d_data"],
                    order_mid=compute["order_mid"],
                    order_last=compute["order_last"],
                    trunc_num=compute["trunc_num"],
                    log_space=compute["log_space"],
                    use_remat=False,
                ),
                "batchscan_nll_remat": make_addm_nll_function_batchscan(
                    dataset["rt_data"],
                    dataset["choice_data"],
                    dataset["r1_data"],
                    dataset["r2_data"],
                    dataset["flag_data"],
                    dataset["sacc_array_data"],
                    dataset["d_data"],
                    order_mid=compute["order_mid"],
                    order_last=compute["order_last"],
                    trunc_num=compute["trunc_num"],
                    log_space=compute["log_space"],
                    use_remat=True,
                ),
            }

            for variant_name, fn in factories.items():
                timed = benchmark_jax_scalar_objective(
                    jax,
                    fn,
                    params,
                    n_calls=n_calls,
                    repeat=1 if args.smoke else 2,
                    grad_argnums=(0, 1, 2, 3, 4, 5),
                )
                record = {
                    "variant": variant_name,
                    "api_kind": "nll_factory",
                    "sweep": {"name": sweep_name, "value": sweep_value},
                    "workload": workload,
                    "compute": {
                        **compute,
                        "n_calls": n_calls,
                        "use_remat": variant_name.endswith("remat"),
                    },
                    "metrics": make_metrics(
                        compile_forward_s=timed["compile"]["forward_s"],
                        compile_value_and_grad_s=timed["compile"]["value_and_grad_s"],
                        runtime_forward_s=timed["runtime"]["forward_s"],
                        runtime_value_and_grad_s=timed["runtime"]["value_and_grad_s"],
                        backward_proxy_s=timed["runtime"]["backward_proxy_s"],
                        units_per_s_value=units_per_second(
                            workload["n_trials"], timed["runtime"]["forward_s"]
                        ),
                        grad_units_per_s_value=units_per_second(
                            workload["n_trials"], timed["runtime"]["value_and_grad_s"]
                        ),
                        forward_memory=timed["memory"]["forward"],
                        value_and_grad_memory=timed["memory"]["value_and_grad"],
                    ),
                    "values": {
                        "result": summarize_value(timed["value_and_grad"][0]),
                        "gradient": summarize_gradient_tree(timed["value_and_grad"][1]),
                    },
                }
                records.append(record)
                print(
                    f"{variant_name:20s}  {str(sweep_value):>10s}  "
                    f"{record['metrics']['compile']['total_s']:10.4f}  "
                    f"{record['metrics']['runtime']['forward_s'] * 1e3:10.3f}  "
                    f"{record['metrics']['runtime']['value_and_grad_s'] * 1e3:10.3f}  "
                    f"{record['metrics']['runtime']['backward_proxy_s'] * 1e3:14.3f}  "
                    f"{format_rate(record['metrics']['runtime']['units_per_s']):>10s}  "
                    f"{format_kib(record['metrics']['memory']['value_and_grad']['temp_kib']):>10s}"
                )

    write_json_output(
        args.output_json,
        benchmark="batch_gpu_scaling",
        records=records,
        args=args,
        jax_module=jax,
    )


if __name__ == "__main__":
    main()
