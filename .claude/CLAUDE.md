# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project

bartz (BART vectoriZed) — a fast implementation of Bayesian Additive Regression Trees (BART) in JAX. Trees are stored as heap arrays for efficient vectorized operations via `jit`/`vmap`/`lax.scan`.

## Commands

All our development commands are make targets. All make targets use `uv run` under the hood.

## Directory layout

We often use a worktree-first layout where the directory bartz/ is not a worktree, while each subdirectory bartz/foo, bartz/bar, etc. is a worktree. When started from a subdirectory, stay there, and don't try to read/access stuff outside of the worktree.

## Workflow

To check the code you write:
- `make lint`
    - cheap to run, unleashes all linters on everything
    - don't show your work without this first!
- when changing/writing documentation for public stuff:
    - run `make docs`
    - check the html documentation is fine
        - the stuff that breaks most often is type hints, return ones in particular
- `make setup` if needed
    - this sets R up and checks python and jax work fine (incl. if jax picks up any gpu)
    - cheap to run (cached), does not clean, idempotent
    - use liberally if it looks like R is not working
- run the unit tests relevant to your code changes with `uv run pytest ...`
    - not all tests right away because the full test suite takes a long time to run
- at the end of debugging, run the full test suite to check everything works
    - `make tests` will run everything with 2 workers and prints a lot of stuff, faster and more complete than `uv run pytest`

## Architecture

**Source layout:** `src/bartz/`

| Module | Role |
|---|---|
| `_interface.py` | `Bart` class — high-level public API |
| `BART/` | R BART3-compatible wrappers (`mc_gbart`, `gbart`). The purpose of `mc_gbart` is to maintain a stable interface matching the R BART3 package; when modifying the library internals, adapt `mc_gbart`'s implementation to fit while preserving its external interface. |
| `stochtree/` | stochtree-compatible wrapper `BARTModel`, stable compatibility interface like BART3 |
| `mcmcstep/` | MCMC state (`State`, `Forest`, `StepConfig`), `init`, `step` |
| `mcmcloop/` | MCMC loop orchestration (`run_mcmc`, `evaluate_trace`) |
| `grove/` | Decision tree operations on heap arrays (leaves, splits, traversal) |
| `prepcovars/` | Covariate preprocessing (binning, standardization, R format parsing) |
| `_jaxext/` | JAX utility extensions (vmap, dtypes, device helpers, `scipy/` subpackage) |
| `debug/` | Trace validation, prior sampling, R↔bartz tree conversion |
| `testing/` | `DGP` and `gen_data` for synthetic datasets |

**Data flow:** covariates → `prepcovars` → `mcmcstep.init` → `mcmcloop.run_mcmc` (calls `mcmcstep.step` per iteration) → `RunMCMCResult` with posterior tree samples

State objects are immutable `equinox.Module` dataclasses. Multi-device parallelism via `jax.sharding`.

Interface hierarchy:
- compatibility wrappers `mc_gbart`, `gbart`, `BARTModel`
    - main user interface `Bart`
        - MCMC setup `init()`, MCMC runner `run_mcmc()`
            - MCMC step `step()`

## Code style

- **Formatter/linter:** ruff with single quotes
- **Imports:** generally use `from foo import bar` (relative import) instead of `import foo; foo.bar`
    - but for some heavily used big (sub)modules, e.g., `from jax import random; random.foo` is preferred to `from jax.random import foo, foo1, foo2, ..., foo999999`.
- **Headers** All source files carry an MIT copyright header
- **docstrings:**
    - numpy convention
    - class attributes documented individually with string just below (not in class docstring)
        - but not global variables
    - keep docstrings short, don't fill them with implementation details
        - related: no redundant comments, if the code is readable, it's self-documenting
        - docstrings and comments shall be _timeless_, not a narration of the development work
        - again, BE BRIEF, humans read slower than you!
    - keep private/internal docstrings short or absent if they are so already
    - keep return value description relatively short and strictly on one line, html render garbles it otherwise
    - do _not_ use colons `:` on the first line of a docstring, it trips napoleon
- **jax** conventions:
    - don't cast to jax arrays things which are already jax arrays (e.g., notice type hints or previous casts)
    - avoid explicit jax types if not necessary, e.g., do `jnp.ones(shape)` instead of `jnp.ones(shape, jnp.float32)`, `1.0` instead of `jnp.float32(1.0)` (unless we need a strong type to auto-cast unsanitized arrays), etc.
        - you can also pass python scalars or numpy arrays to jax-jitted functions and they will be converted
    - try to unify code paths by clever usage of array indexing/broadcasting/axes, e.g., `x[..., i]` will work both if x is 2d or 1d, many places in the library use such tricks
    - indexing/shape conventions that improve readability and implicitly check for shape errors:
        - to get an axis length in an array, use tuple unpacking, e.g.: `_, _, k = x.shape`, `*_, l, _ = y.shape`
        - to index into an array, keep all dimensions explicit, e.g.: `x[0, :]`, `y[..., :, :, 4, :]`
        - to remove an axis of size 1, use `x.squeeze(axis)`, not `x[0]`
            - this implicitly asserts the axis _does_ have size 1
    - use `array.item()` to cast a size-1 array to a scalar python type
        - also works for numpy/jax scalars, `jnp.float32(x).item()` is valid
- other **python** conventions:
    - use dicts as if they were frozendicts when possible: e.g., do `d = dict(d, a=1, b=2)` to set values instead of `d['a'] = 1` or `d.update(a=1)`, safer
        - related: prefer tuples to lists
        - related: make all dataclasses frozen unless you really need mutability
    - type annotations:
        - do not stringify type annotations
        - jaxtyping for array shapes (`Float32[Array, 'n p']`)
            - space before single-axis annotation `Float32[Array, ' n']` because of linter bug
            - don't use '...' to indicate arbitrary shape in _public_ stuff that ends up in the html doc
                - the type html renderer garbles them
                - they are fine and convenient in non-public stuff
        - type hints in signatures, not in docstrings
            - but when returning multiple values, copy the type hints verbatim in the return values list, because the html doc type render does not support multi-valued return natively
    - _src-like layout: modules only contain the public symbols, imported from an implementation submodule
        - because of this, don't prepend redundant underscores to private functions: they stay private
    - prefer `if ...: return; else: return` to early returns
        - if-else block are much easier to read visually for a human, even if redundant due to returns
        - other angle: `return` is a bit like a goto, bad habit to use mid-function
        - of course for some cases it's obviously super-convenient to return early, should be clear when it happens
- **WORKAROUND markers:** we support comments like `# WORKAROUND(jax<99): remove this patch when we bump jax to v99`, enforced by `make lint` checking the oldest supported version of the package
    - also works with python versions
    - also valid for bartz itself, in the context of benchmarking code

## Testing

- pytest, we use parametrization and subtests a lot
- global `keys` fixture provides deterministic per-test JAX random keys (use `keys.pop()`)
    - always use this to produce the seeds, don't hardcode seeds
    - to seed non-jax stuff, do `from tests.util import int_seed; int_seed(keys.pop())`
    - inline `keys.pop()` instead of assigning it to a local---less chance of re-using the key by mistake
    - use `keys.pop(num)` instead of `random.split(keys.pop(), num)`, shortcut
- Custom pytest options: `--platform` (cpu/gpu/auto), `--num-cpu-devices` (sets up jax virtual cpu devices)
- The subpackage `tests/rbartpackages/` contains wrappers of R BART packages, not unit tests
- To compare vectors/matrices/tensors, use `tests.util.assert_close_matrices` instead of numpy's `assert_allclose`
    - use `rtol` in test comparisons, add `atol` only if necessary (comparison of values that are near zero on some relevant scale)
    - there's also `assert_different_matrices` to check things are not equal, this requires to set both atol and rtol which are +inf by default
- in general prefer `assert_` functions from `tests.util` and `numpy.testing` to plain `assert` if appropriate
- use `bartz.testing` utilities to generate data

## Benchmarks

- in `benchmarks/`, import APIs used for scaffolding (utilities) from `benchmarks.latest_bartz` (auto-updated vendored copy), while import from `bartz` (changes version during benchmark run) only the stuff to benchmark
- after editing benchmarks, test them with `make asv-quick ARGS='--bench <pattern>'` or equivalent
