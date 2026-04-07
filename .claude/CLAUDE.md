# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

bartz (BART vectoriZed) — a fast implementation of Bayesian Additive Regression Trees (BART) in JAX. Trees are stored as heap arrays for efficient vectorized operations via `jit`/`vmap`/`lax.scan`.

## Commands

```bash
uv run pytest                     # run all tests
uv run pytest -k test_name        # run a single test or subset
uv run pytest tests/test_foo.py   # run one test file
make tests                        # run tests on CPU with coverage + xdist (CI-like)
make tests-gpu                    # run tests on GPU
make lint                         # ruff check via pre-commit
make docs                         # build Sphinx HTML docs
make setup                        # initial dev environment (Python + R)
```

All commands use `uv run` under the hood. Tests run with `--import-mode=importlib`, 512s timeout. Coverage thresholds: 99% for tests, 90% for src.

## Workflow

- After implementing a feature or fix, run all unit tests.
- Use `make lint` to run `ruff check --fix` in a project-compatible way. Run it to check and auto-fix linting issues.
- Prefer `uv run pytest` over `make tests` for day-to-day testing (simpler output).
- When running all tests, pipe output to a scratch file (e.g., `uv run pytest > scratch_test_output.txt 2>&1`) with a 20-minute timeout, then read it, to avoid flooding the terminal.
- If tests fail with errors unrelated to your changes (missing packages, broken environment), run `make setup` yourself first, then retry. Ignore any failed pre-commit installation during setup.

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
- R integration tests in `tests/rbartpackages/` compare against R BART3 package
