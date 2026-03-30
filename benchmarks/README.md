# Benchmarks

This folder contains the local benchmark suite for `efficient-fpt`.

The scripts are organized by performance question rather than by backend:

- `micro_logsumexp.py`
  - micro-benchmark for the NumPy multistage `_logsumexp` helper
- `single_trial_backends.py`
  - single-trial forward-only comparison across Cython and JAX public APIs
- `single_trial_jax_methods.py`
  - JAX single-trial implementation comparison
  - compares precomputed vs stagescan
  - reports forward/value-and-grad compile and runtime, backward proxy, throughput, and XLA memory
- `batch_gpu_methods.py`
  - recommended starting point for GPU gradient-based MCMC work
  - compares public batch likelihood APIs and public NLL factories
  - includes remat-enabled variants of the public batch NLL factories
- `batch_gpu_scaling.py`
  - scaling study for the production JAX batch MCMC path
  - sweeps `n_trials`, `max_d`, `order`, `trunc_num`, `log_space`, and `precision`

## Shared JSON Schema

Every script writes the same top-level JSON document:

- `benchmark_schema_version`
- `benchmark`
- `created_at`
- `system`
- `arguments`
- `records`

Each record uses the same core shape:

- `variant`
- `api_kind`
- `workload`
- `compute`
- `metrics`
- `values` when applicable

The `metrics` block is nested:

- `compile`
  - `forward_s`
  - `value_and_grad_s` when applicable
  - `total_s`
- `runtime`
  - `forward_s`
  - `value_and_grad_s` when applicable
  - `backward_proxy_s` when applicable
  - `units_per_s` when applicable
  - `grad_units_per_s` when applicable
- `memory`
  - `forward`
  - `value_and_grad` when applicable

## Metric Glossary

- `forward_s`
  - steady-state forward runtime after compilation
- `value_and_grad_s`
  - steady-state runtime for the full scalar objective plus gradient computation
- `backward_proxy_s`
  - `max(value_and_grad_s - forward_s, 0)`
  - an explicit proxy for backward cost, not a standalone reverse-pass measurement
- `units_per_s`
  - throughput in evaluations/s or trials/s, depending on the script
- `grad_units_per_s`
  - throughput for scalar objective + gradient runs
- `memory.forward` / `memory.value_and_grad`
  - XLA `memory_analysis()` results in KiB
  - this suite does not report runtime peak GPU memory in this pass

## Common CLI

All scripts support:

- `--smoke`
  - tiny configuration for import/smoke checks
- `--output-json PATH`
  - write structured JSON results for later comparison

JAX scripts also support a precision selector:

- `--precision`
  - `float32`, `float64`, or `both` where the script supports precision comparison

Default `pytest` runs now exclude the subprocess benchmark smoke checks. To run
those explicitly, use:

`python -m pytest -q -m benchmark_smoke tests/test_benchmarks.py`

## Recommended Usage

For GPU gradient-based MCMC, start with:

1. `python benchmarks/batch_gpu_methods.py`
2. `python benchmarks/batch_gpu_scaling.py`

The other scripts provide context for micro-kernel behavior and single-trial
implementation tradeoffs.
