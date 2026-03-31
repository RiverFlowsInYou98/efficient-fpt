"""Single-trial JAX method benchmark for precomputed vs stagescan kernels."""

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
    make_single_trial_test_data_jax,
    make_metrics,
    resolve_cli_quadrature_orders,
    resolve_precision_values,
    summarize_gradient_tree,
    summarize_value,
    units_per_second,
    write_json_output,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(parser)
    add_jax_precision_argument(parser, allow_both=True, default="both")
    add_quadrature_order_arguments(parser)
    parser.add_argument(
        "--d-values",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10],
        help="Stage counts to benchmark.",
    )
    parser.add_argument(
        "--trunc-num",
        type=int,
        default=10,
        help="Fixed single-stage truncation length.",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=10,
        help="Number of timing calls after compilation.",
    )
    args = parser.parse_args()
    args.order_mid, args.order_last = resolve_cli_quadrature_orders(args)
    return args


def build_variants(jnp, data, order_mid: int, order_last: int, trunc_num: int):
    from efficient_fpt.jax.multi_stage import (
        compute_addm_logfptd_precomputed,
        compute_addm_logfptd_stagescan,
        compute_heterog_multistage_logfptd_precomputed,
        compute_heterog_multistage_logfptd_stagescan,
    )

    t = data["t"]
    d = data["d"]
    mu_arr = data["mu_arr"]
    node_arr = data["node_arr"]
    x0 = data["x0"]
    sigma_arr = data["sigma_arr"]
    b1_arr = data["b1_arr"]
    b2_arr = data["b2_arr"]
    r1 = data["r1"]
    r2 = data["r2"]
    flag = data["flag"]

    variants = {}
    for log_space, log_tag in ((False, "normal"), (True, "log")):
        variants[f"addm/precomputed/{log_tag}"] = lambda s, a_, b_, _ls=log_space: compute_addm_logfptd_precomputed(
            t,
            1,
            0.0,
            1.0,
            s,
            a_,
            b_,
            x0,
            r1,
            r2,
            flag,
            node_arr,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=_ls,
        )
        variants[f"addm/stagescan/{log_tag}"] = lambda s, a_, b_, _ls=log_space: compute_addm_logfptd_stagescan(
            t,
            1,
            0.0,
            1.0,
            s,
            a_,
            b_,
            x0,
            r1,
            r2,
            flag,
            node_arr,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=_ls,
        )
        variants[f"multi/precomputed/{log_tag}"] = lambda s, a_, b_, _ls=log_space, _shape=sigma_arr.shape[0]: compute_heterog_multistage_logfptd_precomputed(
            t,
            1,
            x0,
            a_,
            -a_,
            mu_arr,
            node_arr,
            jnp.full(_shape, s),
            jnp.full(_shape, -b_),
            jnp.full(_shape, b_),
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=_ls,
        )
        variants[f"multi/stagescan/{log_tag}"] = lambda s, a_, b_, _ls=log_space, _shape=sigma_arr.shape[0]: compute_heterog_multistage_logfptd_stagescan(
            t,
            1,
            x0,
            a_,
            -a_,
            mu_arr,
            node_arr,
            jnp.full(_shape, s),
            jnp.full(_shape, -b_),
            jnp.full(_shape, b_),
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=_ls,
        )
    return variants


def main():
    args = parse_args()
    d_values = [2] if args.smoke else args.d_values
    n_calls = 2 if args.smoke else args.n_calls
    precision_values = resolve_precision_values(args.precision, smoke=args.smoke)

    records = []
    print("=" * 104)
    print("SINGLE-TRIAL JAX METHOD BENCHMARKS")
    print("=" * 104)

    for precision in precision_values:
        jax, jnp = import_jax(precision)
        for d in d_values:
            data = make_single_trial_test_data_jax(jnp, d)
            params = (data["sigma"], data["a"], data["b"])

            print(
                f"\n  precision={precision}  d={d}  "
                f"(order_mid={args.order_mid}, order_last={args.order_last}, "
                f"trunc_num={args.trunc_num}, calls={n_calls})"
            )
            print(
                f"  {'variant':25s}  {'compile(s)':>10s}  {'fwd(us)':>10s}  "
                f"{'v+g(us)':>10s}  {'bwd_proxy(us)':>14s}  {'eval/s':>10s}  "
                f"{'grad/s':>10s}  {'fwd temp':>8s}  {'vg temp':>8s}  {'result':>18s}"
            )

            for name, fn in build_variants(
                jnp,
                data,
                args.order_mid,
                args.order_last,
                args.trunc_num,
            ).items():
                timed = benchmark_jax_scalar_objective(
                    jax,
                    fn,
                    params,
                    n_calls=n_calls,
                    repeat=2 if not args.smoke else 1,
                    grad_argnums=(0, 1, 2),
                )
                value, grads = timed["value_and_grad"]
                record = {
                    "variant": name,
                    "api_kind": "scalar_objective",
                    "workload": {"d": d},
                    "compute": {
                        "order_mid": args.order_mid,
                        "order_last": args.order_last,
                        "trunc_num": args.trunc_num,
                        "n_calls": n_calls,
                        "precision": precision,
                    },
                    "metrics": make_metrics(
                        compile_forward_s=timed["compile"]["forward_s"],
                        compile_value_and_grad_s=timed["compile"]["value_and_grad_s"],
                        runtime_forward_s=timed["runtime"]["forward_s"],
                        runtime_value_and_grad_s=timed["runtime"]["value_and_grad_s"],
                        backward_proxy_s=timed["runtime"]["backward_proxy_s"],
                        units_per_s_value=units_per_second(1, timed["runtime"]["forward_s"]),
                        grad_units_per_s_value=units_per_second(
                            1, timed["runtime"]["value_and_grad_s"]
                        ),
                        forward_memory=timed["memory"]["forward"],
                        value_and_grad_memory=timed["memory"]["value_and_grad"],
                    ),
                    "values": {
                        "result": summarize_value(value),
                        "gradient": summarize_gradient_tree(grads),
                    },
                }
                records.append(record)
                print(
                    f"  {name:25s}  {record['metrics']['compile']['total_s']:10.4f}  "
                    f"{timed['runtime']['forward_s'] * 1e6:10.1f}  "
                    f"{timed['runtime']['value_and_grad_s'] * 1e6:10.1f}  "
                    f"{timed['runtime']['backward_proxy_s'] * 1e6:14.1f}  "
                    f"{format_rate(units_per_second(1, timed['runtime']['forward_s'])):>10s}  "
                    f"{format_rate(units_per_second(1, timed['runtime']['value_and_grad_s'])):>10s}  "
                    f"{format_kib(timed['memory']['forward']['temp_kib']):>8s}  "
                    f"{format_kib(timed['memory']['value_and_grad']['temp_kib']):>8s}  "
                    f"{float(value):18.10e}"
                )

    write_json_output(
        args.output_json,
        benchmark="single_trial_jax_methods",
        records=records,
        args=args,
        jax_module=jax if precision_values else None,
    )


if __name__ == "__main__":
    main()
