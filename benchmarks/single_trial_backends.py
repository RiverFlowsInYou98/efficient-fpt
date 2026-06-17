"""Single-trial forward benchmark across Cython and JAX backends."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

from benchmarks.common import (
    add_common_cli_arguments,
    add_jax_precision_argument,
    add_quadrature_order_arguments,
    benchmark_jax_forward_callable,
    best_time,
    format_kib,
    format_rate,
    import_jax,
    make_single_trial_test_data_jax,
    make_single_trial_test_data_np,
    make_metrics,
    resolve_cli_quadrature_orders,
    summarize_value,
    units_per_second,
    write_json_output,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(parser)
    add_jax_precision_argument(parser, default="float64")
    add_quadrature_order_arguments(parser)
    parser.add_argument(
        "--d-values",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10],
        help="Stage counts to benchmark.",
    )
    parser.add_argument(
        "--cython-calls",
        type=int,
        default=5000,
        help="Number of timing calls for the Cython variants.",
    )
    parser.add_argument(
        "--jax-calls",
        type=int,
        default=20,
        help="Number of timing calls for the JAX variants after compilation.",
    )
    parser.add_argument(
        "--trunc-num",
        type=int,
        default=10,
        help="Fixed single-stage truncation length for JAX public APIs.",
    )
    args = parser.parse_args()
    args.order_mid, args.order_last = resolve_cli_quadrature_orders(args)
    return args


def build_jax_variants(jnp, data, order_mid: int, order_last: int, trunc_num: int):
    from efpt.jax.multi_stage import (
        compute_addm_logfptd_precomputed,
        compute_addm_logfptd_stagescan,
        compute_heterog_multistage_logfptd_precomputed,
        compute_heterog_multistage_logfptd_stagescan,
    )

    t = data["t"]
    d = data["d"]
    mu_arr = data["mu_arr"]
    node_arr = data["node_arr"]
    sigma = data["sigma"]
    a = data["a"]
    b = data["b"]
    x0 = data["x0"]
    sigma_arr = data["sigma_arr"]
    b1_arr = data["b1_arr"]
    b2_arr = data["b2_arr"]
    r1 = data["r1"]
    r2 = data["r2"]
    flag = data["flag"]

    variants = {}
    for log_space, log_tag in ((False, "normal"), (True, "log")):
        variants[f"jax/addm/precomputed/{log_tag}"] = lambda s, a_, b_, _ls=log_space: compute_addm_logfptd_precomputed(
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
        variants[f"jax/addm/stagescan/{log_tag}"] = lambda s, a_, b_, _ls=log_space: compute_addm_logfptd_stagescan(
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
        variants[f"jax/multi/precomputed/{log_tag}"] = lambda s, a_, b_, _ls=log_space, _shape=sigma_arr.shape[0]: compute_heterog_multistage_logfptd_precomputed(
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
        variants[f"jax/multi/stagescan/{log_tag}"] = lambda s, a_, b_, _ls=log_space, _shape=sigma_arr.shape[0]: compute_heterog_multistage_logfptd_stagescan(
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
    cython_calls = 50 if args.smoke else args.cython_calls
    jax_calls = 2 if args.smoke else args.jax_calls

    from efpt.cython.multi_stage import (
        compute_addm_logfptd as cython_addm_logfptd,
        compute_heterog_multistage_logfptd as cython_multi_logfptd,
    )

    jax, jnp = import_jax(args.precision)

    records = []
    print("=" * 88)
    print("SINGLE-TRIAL BACKEND BENCHMARKS")
    print("=" * 88)
    print(
        f"{'variant':30s}  {'d':>3s}  {'prec':>7s}  {'compile(s)':>10s}  "
        f"{'fwd(us)':>10s}  {'eval/s':>10s}  {'temp(KiB)':>10s}  {'result':>18s}"
    )

    for d in d_values:
        t, mu, node, sigma, a, b, x0, sigma_arr, b1_arr, b2_arr = make_single_trial_test_data_np(d)
        data_jax = make_single_trial_test_data_jax(jnp, d)

        cython_variants = {
            "cython/addm/normal": lambda: cython_addm_logfptd(
                t,
                1,
                0.0,
                1.0,
                sigma,
                a,
                b,
                x0,
                0.1,
                0.1,
                0,
                node,
                d,
                order_mid=args.order_mid,
                order_last=args.order_last,
                log_space=False,
            ),
            "cython/addm/log": lambda: cython_addm_logfptd(
                t,
                1,
                0.0,
                1.0,
                sigma,
                a,
                b,
                x0,
                0.1,
                0.1,
                0,
                node,
                d,
                order_mid=args.order_mid,
                order_last=args.order_last,
                log_space=True,
            ),
            "cython/multi/normal": lambda: cython_multi_logfptd(
                t,
                1,
                x0,
                a,
                -a,
                mu,
                node,
                sigma_arr,
                b1_arr,
                b2_arr,
                d,
                order_mid=args.order_mid,
                order_last=args.order_last,
                log_space=False,
            ),
            "cython/multi/log": lambda: cython_multi_logfptd(
                t,
                1,
                x0,
                a,
                -a,
                mu,
                node,
                sigma_arr,
                b1_arr,
                b2_arr,
                d,
                order_mid=args.order_mid,
                order_last=args.order_last,
                log_space=True,
            ),
        }

        for name, fn in cython_variants.items():
            result = float(fn())
            forward_s = best_time(fn, n_calls=cython_calls, repeat=3 if not args.smoke else 1)
            records.append(
                {
                    "variant": name,
                    "api_kind": "single_trial_forward",
                    "workload": {"d": d},
                    "compute": {
                        "order_mid": args.order_mid,
                        "order_last": args.order_last,
                        "n_calls": cython_calls,
                        "trunc_num": None,
                        "precision": "float64",
                    },
                    "metrics": make_metrics(
                        runtime_forward_s=forward_s,
                        units_per_s_value=units_per_second(1, forward_s),
                    ),
                    "values": {"result": summarize_value(result)},
                }
            )
            print(
                f"{name:30s}  {d:3d}  {'float64':>7s}  {'-':>10s}  "
                f"{forward_s * 1e6:10.1f}  {format_rate(units_per_second(1, forward_s)):>10s}  "
                f"{'N/A':>10s}  {result:18.10e}"
            )

        for name, fn in build_jax_variants(
            jnp, data_jax, args.order_mid, args.order_last, args.trunc_num
        ).items():
            params = (data_jax["sigma"], data_jax["a"], data_jax["b"])
            timed = benchmark_jax_forward_callable(
                jax,
                fn,
                params,
                n_calls=jax_calls,
                repeat=2 if not args.smoke else 1,
            )
            result = float(timed["forward_value"])
            records.append(
                {
                    "variant": name,
                    "api_kind": "single_trial_forward",
                    "workload": {"d": d},
                    "compute": {
                        "order_mid": args.order_mid,
                        "order_last": args.order_last,
                        "n_calls": jax_calls,
                        "trunc_num": args.trunc_num,
                        "precision": args.precision,
                    },
                    "metrics": make_metrics(
                        compile_forward_s=timed["compile"]["forward_s"],
                        runtime_forward_s=timed["runtime"]["forward_s"],
                        units_per_s_value=units_per_second(1, timed["runtime"]["forward_s"]),
                        forward_memory=timed["memory"]["forward"],
                    ),
                    "values": {"result": summarize_value(result)},
                }
            )
            print(
                f"{name:30s}  {d:3d}  {args.precision:>7s}  "
                f"{timed['compile']['forward_s']:10.4f}  "
                f"{timed['runtime']['forward_s'] * 1e6:10.1f}  "
                f"{format_rate(units_per_second(1, timed['runtime']['forward_s'])):>10s}  "
                f"{format_kib(timed['memory']['forward']['temp_kib']):>10s}  "
                f"{result:18.10e}"
            )

    write_json_output(
        args.output_json,
        benchmark="single_trial_backends",
        records=records,
        args=args,
        jax_module=jax,
    )


if __name__ == "__main__":
    main()
