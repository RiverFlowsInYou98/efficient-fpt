"""GPU-oriented batch JAX method benchmark for gradient-based inference."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from benchmarks.common import (
    add_common_cli_arguments,
    add_jax_precision_argument,
    benchmark_jax_forward_callable,
    benchmark_jax_scalar_objective,
    format_kib,
    format_rate,
    import_jax,
    make_batch_addm_dataset,
    make_metrics,
    resolve_precision_values,
    summarize_gradient_tree,
    summarize_value,
    units_per_second,
    write_json_output,
)


DEFAULT_PARAMS = (0.25, 1.1, 1.0, 1.5, 0.3, 0.0)
DEFAULT_WORKLOADS = [
    ("small", 128, 4),
    ("medium", 512, 8),
    ("large", 1024, 12),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(parser)
    add_jax_precision_argument(parser, allow_both=True, default="both")
    parser.add_argument("--order", type=int, default=30, help="Quadrature order.")
    parser.add_argument(
        "--trunc-num",
        type=int,
        default=50,
        help="Fixed single-stage truncation length.",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=10,
        help="Number of timing calls after compilation.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    n_calls = 1 if args.smoke else args.n_calls
    workloads = [("smoke", 16, 3)] if args.smoke else DEFAULT_WORKLOADS
    log_space_values = (False,) if args.smoke else (False, True)
    precision_values = resolve_precision_values(args.precision, smoke=args.smoke)
    jax = None

    records = []
    print("=" * 96)
    print("BATCH GPU METHOD BENCHMARKS")
    print("=" * 96)
    for precision in precision_values:
        jax, jnp = import_jax(precision)
        from efficient_fpt.jax.batch import (
            compute_addm_likelihoods_batchscan,
            compute_addm_likelihoods_batchvmap,
            make_addm_nll_function_batchscan,
            make_addm_nll_function_batchvmap,
        )

        params = tuple((jnp.float32 if precision == "float32" else jnp.float64)(value) for value in DEFAULT_PARAMS)

        print(f"devices={jax.devices()}")
        for log_space in log_space_values:
            print("\n" + "-" * 112)
            print(f"precision={precision}  log_space={log_space}")
            print("-" * 112)
            print(
                f"{'workload':10s}  {'api kind':14s}  {'variant':24s}  "
                f"{'compile(s)':>10s}  {'fwd(ms)':>10s}  {'v+g(ms)':>10s}  "
                f"{'bwd_proxy(ms)':>14s}  {'trials/s':>10s}  {'temp(KiB)':>10s}"
            )

            for workload_idx, (workload_name, n_trials, max_d) in enumerate(workloads):
                dataset = make_batch_addm_dataset(
                    jnp,
                    n_trials=n_trials,
                    max_d=max_d,
                    seed=1000 + workload_idx + 10 * int(log_space),
                )

                likelihood_variants = {
                    "batchvmap": lambda eta, kappa, sigma, a, b, x0: compute_addm_likelihoods_batchvmap(
                        dataset["rt_data"],
                        dataset["choice_data"],
                        eta,
                        kappa,
                        sigma,
                        a,
                        b,
                        x0,
                        dataset["r1_data"],
                        dataset["r2_data"],
                        dataset["flag_data"],
                        dataset["sacc_array_data"],
                        dataset["d_data"],
                        order=args.order,
                        trunc_num=args.trunc_num,
                        log_space=log_space,
                    ),
                    "batchscan": lambda eta, kappa, sigma, a, b, x0: compute_addm_likelihoods_batchscan(
                        dataset["rt_data"],
                        dataset["choice_data"],
                        eta,
                        kappa,
                        sigma,
                        a,
                        b,
                        x0,
                        dataset["r1_data"],
                        dataset["r2_data"],
                        dataset["flag_data"],
                        dataset["sacc_array_data"],
                        dataset["d_data"],
                        order=args.order,
                        trunc_num=args.trunc_num,
                        log_space=log_space,
                    ),
                }

                for variant_name, fn in likelihood_variants.items():
                    timed = benchmark_jax_forward_callable(
                        jax,
                        fn,
                        params,
                        n_calls=n_calls,
                        repeat=1 if args.smoke else 2,
                    )
                    records.append(
                        {
                            "variant": variant_name,
                            "api_kind": "likelihood",
                            "workload": {
                                "name": workload_name,
                                "n_trials": n_trials,
                                "max_d": max_d,
                            },
                            "compute": {
                                "order": args.order,
                                "trunc_num": args.trunc_num,
                                "log_space": log_space,
                                "n_calls": n_calls,
                                "precision": precision,
                                "use_remat": False,
                            },
                            "metrics": make_metrics(
                                compile_forward_s=timed["compile"]["forward_s"],
                                runtime_forward_s=timed["runtime"]["forward_s"],
                                units_per_s_value=units_per_second(
                                    n_trials, timed["runtime"]["forward_s"]
                                ),
                                forward_memory=timed["memory"]["forward"],
                            ),
                            "values": {
                                "result": summarize_value(timed["forward_value"]),
                            },
                        }
                    )
                    print(
                        f"{workload_name:10s}  {'likelihood':14s}  {variant_name:24s}  "
                        f"{timed['compile']['forward_s']:10.4f}  "
                        f"{timed['runtime']['forward_s'] * 1e3:10.3f}  {'-':>10s}  "
                        f"{'-':>14s}  "
                        f"{format_rate(units_per_second(n_trials, timed['runtime']['forward_s'])):>10s}  "
                        f"{format_kib(timed['memory']['forward']['temp_kib']):>10s}"
                    )

                nll_factories = {
                    "batchvmap_nll": make_addm_nll_function_batchvmap(
                        dataset["rt_data"],
                        dataset["choice_data"],
                        dataset["r1_data"],
                        dataset["r2_data"],
                        dataset["flag_data"],
                        dataset["sacc_array_data"],
                        dataset["d_data"],
                        order=args.order,
                        trunc_num=args.trunc_num,
                        log_space=log_space,
                        use_remat=False,
                    ),
                    "batchvmap_nll_remat": make_addm_nll_function_batchvmap(
                        dataset["rt_data"],
                        dataset["choice_data"],
                        dataset["r1_data"],
                        dataset["r2_data"],
                        dataset["flag_data"],
                        dataset["sacc_array_data"],
                        dataset["d_data"],
                        order=args.order,
                        trunc_num=args.trunc_num,
                        log_space=log_space,
                        use_remat=True,
                    ),
                    "batchscan_nll": make_addm_nll_function_batchscan(
                        dataset["rt_data"],
                        dataset["choice_data"],
                        dataset["r1_data"],
                        dataset["r2_data"],
                        dataset["flag_data"],
                        dataset["sacc_array_data"],
                        dataset["d_data"],
                        order=args.order,
                        trunc_num=args.trunc_num,
                        log_space=log_space,
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
                        order=args.order,
                        trunc_num=args.trunc_num,
                        log_space=log_space,
                        use_remat=True,
                    ),
                }

                for variant_name, fn in nll_factories.items():
                    timed = benchmark_jax_scalar_objective(
                        jax,
                        fn,
                        params,
                        n_calls=n_calls,
                        repeat=1 if args.smoke else 2,
                        grad_argnums=(0, 1, 2, 3, 4, 5),
                    )
                    value, grads = timed["value_and_grad"]
                    records.append(
                        {
                            "variant": variant_name,
                            "api_kind": "nll_factory",
                            "workload": {
                                "name": workload_name,
                                "n_trials": n_trials,
                                "max_d": max_d,
                            },
                            "compute": {
                                "order": args.order,
                                "trunc_num": args.trunc_num,
                                "log_space": log_space,
                                "n_calls": n_calls,
                                "precision": precision,
                                "use_remat": variant_name.endswith("remat"),
                            },
                            "metrics": make_metrics(
                                compile_forward_s=timed["compile"]["forward_s"],
                                compile_value_and_grad_s=timed["compile"]["value_and_grad_s"],
                                runtime_forward_s=timed["runtime"]["forward_s"],
                                runtime_value_and_grad_s=timed["runtime"]["value_and_grad_s"],
                                backward_proxy_s=timed["runtime"]["backward_proxy_s"],
                                units_per_s_value=units_per_second(
                                    n_trials, timed["runtime"]["forward_s"]
                                ),
                                grad_units_per_s_value=units_per_second(
                                    n_trials, timed["runtime"]["value_and_grad_s"]
                                ),
                                forward_memory=timed["memory"]["forward"],
                                value_and_grad_memory=timed["memory"]["value_and_grad"],
                            ),
                            "values": {
                                "result": summarize_value(value),
                                "gradient": summarize_gradient_tree(grads),
                            },
                        }
                    )
                    print(
                        f"{workload_name:10s}  {'nll_factory':14s}  {variant_name:24s}  "
                        f"{(timed['compile']['forward_s'] + timed['compile']['value_and_grad_s']):10.4f}  "
                        f"{timed['runtime']['forward_s'] * 1e3:10.3f}  "
                        f"{timed['runtime']['value_and_grad_s'] * 1e3:10.3f}  "
                        f"{timed['runtime']['backward_proxy_s'] * 1e3:14.3f}  "
                        f"{format_rate(units_per_second(n_trials, timed['runtime']['forward_s'])):>10s}  "
                        f"{format_kib(timed['memory']['value_and_grad']['temp_kib']):>10s}"
                    )

    write_json_output(
        args.output_json,
        benchmark="batch_gpu_methods",
        records=records,
        args=args,
        jax_module=jax,
    )


if __name__ == "__main__":
    main()
