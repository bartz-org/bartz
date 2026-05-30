# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project

bartz (BART vectoriZed) — a fast implementation of Bayesian Additive Regression Trees (BART) in JAX. Trees are stored as heap arrays for efficient vectorized operations via `jit`/`vmap`/`lax.scan`.

## Commands

```bash
make setup                        # setup dev env (Python and, more importantly, R)
uv run pytest                     # run all tests
make tests                        # run all tests but faster (use parallelization) + very verbose configs
uv run pytest -k test_name        # run a single test or subset
uv run pytest tests/test_foo.py   # run one test file
uv run pytest --lf                # run the tests failed last time
make lint                         # run all linters on everything
make docs                         # build Sphinx HTML docs
```

All make targets use `uv run` under the hood.

## Directory layout

We often use a worktree-first layout where the directory bartz/ is not a worktree, while each subdirectory bartz/foo, bartz/bar, etc. is a worktree. When started from a subdirectory, stay there, and don't try to read/access stuff outside of the worktree.

## Workflow

To check the code you write:
- `make lint`
- when adding/removing/moving functions or classes:
    - check the rst documentation in `docs/` is up-to-date
    - run `make docs`
- `make setup`
    - this sets R up if you are in a clean worktree
- run the unit tests relevant to your code changes with `uv run pytest ...`
    - not all tests right away because the full test suite takes a long time to run
    - when running multiple tests, the output may be long; pipe the output to a scratch file to be read afterwards
    - use `scratch_tests_output.txt` for temporary test output, that specific file name is gitignored just for you
- at the end of debugging, run the full test suite to check everything works
    - use the command `make tests > scratch_tests_output.txt 2>&1` _verbatim_, it's pre-authorized in your configs

## Architecture

**Source layout:** `src/bartz/`

| Module | Role |
|---|---|
| `_interface.py` | `Bart` class — high-level public API |
| `BART/` | R BART3-compatible wrappers (`mc_gbart`, `gbart`). The purpose of `mc_gbart` is to maintain a stable interface matching the R BART3 package; when modifying the library internals, adapt `mc_gbart`'s implementation to fit while preserving its external interface. |
| `mcmcstep/` | MCMC state (`State`, `Forest`, `StepConfig`), `init`, `step` |
| `mcmcloop.py` | MCMC loop orchestration (`run_mcmc`, `evaluate_trace`) |
| `grove.py` | Decision tree operations on heap arrays (leaves, splits, traversal) |
| `prepcovars.py` | Covariate preprocessing (binning, standardization, R format parsing) |
| `jaxext/` | JAX utility extensions (vmap, dtypes, device helpers, `scipy/` subpackage) |
| `debug/` | Trace validation, prior sampling, R↔bartz tree conversion |
| `testing/` | `DGP` and `gen_data` for synthetic datasets |

**Data flow:** covariates → `prepcovars` → `mcmcstep.init` → `mcmcloop.run_mcmc` (calls `mcmcstep.step` per iteration) → `RunMCMCResult` with posterior tree samples.

State objects are immutable `equinox.Module` dataclasses. Multi-device parallelism via `jax.sharding`.

## Code style

- **Formatter/linter:** ruff with single quotes
- **Imports:** generally use `from foo import bar` (relative import) instead of `import foo; foo.bar`, but for some heavily used big (sub)modules, e.g., `from jax import random; random.foo` is preferred to `from jax.random import foo, foo1, foo2, ..., foo999999`.
- **Headers** All source files carry an MIT copyright header
- **docstrings:**
    - numpy convention
    - class attributes documented individually with string just below (not in class docstring)
    - keep docstrings short, don't fill them with implementation details
        - related: no redundant comments, if the code is readable, it's self-documenting
    - keep private/internal docstrings short or absent if they are so already
    - keep return value description relatively short and strictly on one line, html render garbles it otherwise
- **jax** conventions:
    - don't cast to jax arrays things which are already jax arrays (e.g., notice type hints or previous casts)
    - avoid explicit jax types if not necessary, e.g., do `jnp.ones(shape)` instead of `jnp.ones(shape, jnp.float32)`, `1.0` instead of `jnp.float32(1.0)` (unless we need a strong type to auto-cast unsanitized arrays), etc.
        - you can also pass python scalars or numpy arrays to jax-jitted functions and they will be converted
    - try to unify code paths by clever usage of array indexing/broadcasting/axes, e.g., `x[..., i]` will work both if x is 2d or 1d, many places in the library use such tricks
    - indexing/shape conventions that improve readability and implicitly check for shape errors:
        - to get an axis length in an array, use tuple unpacking, e.g.: `_, _, k = x.shape`, `*_, l, _ = y.shape`
        - to index into an array, keep all dimensions explicit, e.g.: `x[0, :]`, `y[..., :, :, 4, :]`
    - use `array.item()` to cast an array to a scalar python type
- other **python** conventions:
    - use dicts as if they were frozendicts when possible: e.g., do `d = dict(d, a=1, b=2)` to set values instead of `d['a'] = 1` or `d.update(a=1)`, safer
        - related: prefer tuples to lists
    - type annotations:
        - do not stringify type annotations
        - jaxtyping for array shapes (`Float32[Array, 'n p']`)
            - space before single-axis annotation `Float32[Array, ' n']` because of linter bug
        - type hints in signatures, not in docstrings
            - but when returning multiple values, copy the type hints verbatim in the return values list, because the html doc render does not support multi-valued return natively
- **WORKAROUND markers:** we support comments like `# WORKAROUND(jax<99): remove this patch when we bump jax to v99`, enforced by `make lint` checking the oldest supported version of the package, also works with python versions

## Testing

- Framework: pytest, we also use subtests and heavily use parametrization
- global `keys` fixture provides deterministic per-test JAX random keys (use `keys.pop()`)
- Custom pytest options: `--platform` (cpu/gpu/auto), `--num-cpu-devices` (sets up jax virtual cpu devices)
- The subpackage `tests/rbartpackages/` contains wrappers of R BART packages, not unit tests
- To compare vectors/matrices/tensors, use `tests.util.assert_close_matrices` instead of numpy's `assert_allclose`
    - use `rtol` in test comparisons, add `atol` only if necessary (comparison of values that are near zero on some relevant scale)
    - there's also `assert_different_matrices` to check things are not equal, this requires to set both atol and rtol which are +inf by default
- in general prefer `assert_` functions from `tests.util` and `numpy.testing` to plain `assert` if appropriate

## Benchmarks

- in `benchmarks/`, import APIs used for scaffolding (utilities) from `benchmarks.latest_bartz` (auto-updated vendored copy), while import from `bartz` (changes version during benchmark run) only the stuff to benchmark
- after editing benchmarks, test them with `make asv-quick ARGS='--bench <pattern>'` or equivalent
