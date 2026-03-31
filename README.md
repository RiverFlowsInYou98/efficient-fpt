# efficient-fpt

`efficient-fpt` implements fast numerical methods for first-passage time
densities in first passage time models, with a particular focus on
likelihood-based inference for generalized drift diffusion
models (GDDMs) in computational cognitive neuroscience.

This repository contains the package code, benchmark suite, and example
notebooks accompanying the paper
[*Efficient Inference in First Passage Time Models*](https://www.arxiv.org/abs/2503.18381).

## What This Package Provides

- Fast Cython implementations for likelihood evaluation and simulation
- JAX implementations for GPU execution and automatic differentiation
- Model classes for homogeneous/heterogeneous DDMs, including multistage models and attentional DDMs
- Batch log-likelihood / NLL APIs for fitting many aDDM trials at once
- Example notebooks for simulation, approximation, inference, and benchmarking

## Backend Overview

The package is organized around complementary backends:

- `efficient_fpt.cython`
  - fastest CPU-oriented production path for many likelihood and simulation
    workloads
- `efficient_fpt.jax`
  - JAX-native kernels for GPU execution and gradient-based inference
- `efficient_fpt.numpy`
  - reference-style NumPy implementations and supporting utilities, convenient log-density computation for homogeneous DDMs

<!-- For JAX there are explicit implementation families:

- single-trial precomputed kernels
  - `compute_addm_logfptd_precomputed`
  - `compute_heterog_multistage_logfptd_precomputed`
- single-trial stage-scan kernels
  - `compute_addm_logfptd_stagescan`
  - `compute_heterog_multistage_logfptd_stagescan`
- batch aDDM kernels
  - `compute_addm_loglikelihoods_batchscan`
  - `compute_addm_loglikelihoods_batchvmap`
- batch NLL factories
  - `make_addm_nll_function_batchscan`
  - `make_addm_nll_function_batchvmap`

The plain default aliases point to the preferred production methods:

- `compute_addm_logfptd == compute_addm_logfptd_precomputed`
- `compute_heterog_multistage_logfptd == compute_heterog_multistage_logfptd_precomputed`
- `compute_addm_loglikelihoods == compute_addm_loglikelihoods_batchscan`
- `make_addm_nll_function == make_addm_nll_function_batchscan` -->

## Installation

### Basic editable install

```bash
git clone git@github.com:RiverFlowsInYou98/efficient-fpt.git
cd efficient-fpt
python -m pip install -e .
```

This project includes Cython extensions. A normal editable install will build
them during installation.

### Optional extras

```bash
python -m pip install -e ".[jax]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[pymc]"
python -m pip install -e ".[test]"
python -m pip install -e ".[all]"
```

### Clean rebuild of Cython extensions

If you are actively changing Cython code and want a clean rebuild:

```bash
bash rebuild.sh
```

## Quick Start

### 1. Model-based workflow

For many CPU workflows, the model classes are the easiest entry point.

```python
import numpy as np
from efficient_fpt import aDDModel

model = aDDModel(
    eta=0.25,
    kappa=1.1,
    sigma=1.0,
    a=1.5,
    b=0.3,
    x0=0.0,
)

r1_data = np.array([0.7, 0.2, 0.6], dtype=float)
r2_data = np.array([0.3, 0.8, 0.4], dtype=float)
flag_data = np.array([0, 1, 0], dtype=np.int32)
sacc_array_data = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.4, 1.1],
    ],
    dtype=float,
)
d_data = np.array([1, 2, 3], dtype=np.int32)

rt, choice, x_final = model.simulate_fpt(
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    T=3.0,
    dt=1e-4,
    rng=0,
)

mean_nll = model.mean_neg_log_likelihood(
    rt,
    choice,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
)
```

This returns a negative log-likelihood directly. Multistage and batch APIs in
the package are log-first; if you need density-scale values for plotting, call
`np.exp(...)` or `jnp.exp(...)` explicitly at the notebook or script boundary.

### 2. JAX batch objective for gradient-based inference

For GPU or autodiff-based inference, use the JAX batch API directly.

```python
import jax
import jax.numpy as jnp
from efficient_fpt.jax import (
    set_jax_precision,
    make_addm_nll_function_batchscan,
)

set_jax_precision(True)  # opt into float64 when desired

rt_data = jnp.array([0.9, 1.5, 2.0])
choice_data = jnp.array([1, -1, 1], dtype=jnp.int32)
r1_data = jnp.array([0.45, 0.30, 0.20])
r2_data = jnp.array([0.10, 0.55, 0.40])
flag_data = jnp.array([0, 1, 0], dtype=jnp.int32)
sacc_array_data = jnp.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.0, 0.0],
        [0.0, 0.6, 1.3, 0.0],
    ]
)
d_data = jnp.array([1, 2, 3], dtype=jnp.int32)

nll_fn = make_addm_nll_function_batchscan(
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    order_mid=30,
    order_last=30,
    trunc_num=50,
    log_space=False,
    use_remat=True,
)

loss = nll_fn(0.25, 1.1, 1.0, 1.5, 0.3, 0.0)
grads = jax.grad(nll_fn, argnums=(0, 1, 2, 3, 4, 5))(0.25, 1.1, 1.0, 1.5, 0.3, 0.0)
```

Notes:

- Importing `efficient_fpt.jax` does not mutate global JAX precision settings.
- Use `set_jax_precision(True)` when you explicitly want float64.
- For gradient-heavy batch workloads, `use_remat=True` can significantly reduce
  reverse-mode memory use.
- `log_space` selects the internal compute mode; the public multistage and batch
  APIs still return log-values in either mode.

<!-- ## Main Public APIs

### Top-level package

```python
from efficient_fpt import (
    DDModel,
    SingleStageModel,
    MultiStageModel,
    aDDModel,
    save_simulation,
    load_simulation,
    save_addm_experiment,
    load_addm_experiment,
)
```

### JAX package

```python
from efficient_fpt.jax import (
    fptd_single,
    q_single,
    log_fptd_single,
    log_q_single,
    compute_addm_logfptd,
    compute_addm_logfptd_precomputed,
    compute_addm_logfptd_stagescan,
    compute_heterog_multistage_logfptd,
    compute_heterog_multistage_logfptd_precomputed,
    compute_heterog_multistage_logfptd_stagescan,
    compute_addm_loglikelihoods,
    compute_addm_loglikelihoods_batchscan,
    compute_addm_loglikelihoods_batchvmap,
    compute_addm_nll,
    make_addm_nll_function,
    make_addm_nll_function_batchscan,
    make_addm_nll_function_batchvmap,
    set_jax_precision,
    get_jax_dtype,
)
``` -->

<!-- ## Examples

Example notebooks live in [examples](/users/sliu167/WorkSpace/efficient-fpt-review/examples). A few useful entry points:

- [examples/example0/example0_single-stage.ipynb](/users/sliu167/WorkSpace/efficient-fpt-review/examples/example0/example0_single-stage.ipynb)
  - single-stage densities and simulation
- [examples/example1/example1_multi-stage.ipynb](/users/sliu167/WorkSpace/efficient-fpt-review/examples/example1/example1_multi-stage.ipynb)
  - multistage densities
- [examples/tutorial_numpy_vs_jax.ipynb](/users/sliu167/WorkSpace/efficient-fpt-review/examples/tutorial_numpy_vs_jax.ipynb)
  - NumPy vs JAX walkthrough
- [examples/inference_comparison.ipynb](/users/sliu167/WorkSpace/efficient-fpt-review/examples/inference_comparison.ipynb)
  - inference-oriented comparison
- [examples/pymc_sampler_comparison.ipynb](/users/sliu167/WorkSpace/efficient-fpt-review/examples/pymc_sampler_comparison.ipynb)
  - PyMC-oriented workflow

Install the `examples` extra if you want to run the notebooks:

```bash
python -m pip install -e ".[examples]"
```

## Benchmarks

Benchmark scripts live in [benchmarks](/users/sliu167/WorkSpace/efficient-fpt-review/benchmarks). The most useful entry points are:

- `python benchmarks/single_trial_backends.py`
- `python benchmarks/single_trial_jax_methods.py`
- `python benchmarks/batch_gpu_methods.py`
- `python benchmarks/batch_gpu_scaling.py`

Smoke / maintenance mode:

```bash
python benchmarks/batch_gpu_methods.py --smoke
```

See [benchmarks/README.md](/users/sliu167/WorkSpace/efficient-fpt-review/benchmarks/README.md) for the full script breakdown and JSON output schema.

## Testing

Run the default test suite:

```bash
python -m pytest -q
```

Benchmark smoke checks are excluded from default pytest runs. To include them:

```bash
python -m pytest -q -m benchmark_smoke tests/test_benchmarks.py
``` -->

## Development Notes

- Cython extensions use OpenMP where available.
- JAX and Cython are intentionally both supported; they optimize for different
  inference regimes.
- Repository-supported persistence uses safe compressed `.npz` archives rather
  than pickle.
- Canonical aDDM experiment I/O is available through:
  - `save_addm_experiment`
  - `load_addm_experiment`
- Generic flat simulation archives are available through:
  - `save_simulation`
  - `load_simulation`

## License

[MIT License](https://opensource.org/licenses/MIT)

## Citation

```bibtex
@article{liu2026efficient,
  title={Efficient inference in first passage time models},
  author={Liu, Sicheng and Fengler, Alexander and Frank, Michael J and Harrison, Matthew T},
  journal={Statistics and Computing},
  volume={36},
  number={3},
  pages={101},
  year={2026},
  publisher={Springer}
}
```
