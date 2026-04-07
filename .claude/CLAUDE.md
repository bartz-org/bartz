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
make lint                         # ruff check --fix, invoked via pre-commit
make docs                         # build Sphinx HTML docs
```

All make targets use `uv run` under the hood.

## Workflow

To check the code you write:
- `make lint`
- `make setup`
    - pre-commit may complain during this if you are in a worktree, ignore that
    - this is needed just once in your session to set up the env to run unit tests
- run the unit tests relevant to your code changes with `uv run pytest ...`
    - not all tests right away because the full test suite takes a long time to run
    - when running multiple tests, the output may be long; pipe the output to a scratch file to be read afterwards
    - use `scratch_tests_output.txt` for temporary test output, that specific file name is gitignored just for you
- at the end of debugging, run the full test suite to check everything works
    - use the command `make tests > make_tests_output.txt 2>&1` _verbatim_, it's pre-authorized in your configs

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

- **Formatter/linter:** ruff with single quotes, numpy docstring convention
- **Imports:** `import jax.numpy as jnp` (enforced alias); `jax.random`, `jax.lax`, `jax.numpy`, `jax.tree` must be imported as modules, not `from`-imported; no relative imports
- **Banned:** `jax.lax.reciprocal` (use `jnp.reciprocal`), `jax.random.PRNGKey` (use `jax.random.key`)
- **Type annotations:** jaxtyping for array shapes (`Float32[Array, 'n p']`), signatures not docstrings
- **All source files** carry an MIT copyright header
- **Docstrings:** numpy convention, enforced by pydoclint; class attributes documented individually (not in class docstring)

## Testing

- Framework: pytest with `pytest-xdist`, `pytest-subtests`, `pytest-timeout`, `flaky`
- `keys` fixture provides deterministic per-test JAX random keys (use `keys.pop()`)
- Debug flags enabled in conftest: `jax_debug_key_reuse`, `jax_debug_nans`, `jax_debug_infs`, `jax_legacy_prng_key='error'`
- Custom options: `--platform` (cpu/gpu/auto), `--num-cpu-devices`
- the subpackage `tests/rbartpackages/` contains wrappers of R BART packages
