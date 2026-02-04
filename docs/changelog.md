<!--
bartz/docs/changelog.md

Copyright (c) 2024-2026, The Bartz Contributors

This file is part of bartz.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

<!-- This changelog is written in Markdown and without line splits to make it
  copy-pastable to github releases. -->


# Changelog


## 0.8.0 The Dark Side of Big Bayes (2026-02-04)

The highlights of this release are multiple parallel Markov chains and splitting data and/or chains across multiple devices. The repository has been moved to [bartz-org](https://github.com/bartz-org) to welcome the first contributor @miaoqingyu2.

* New features
    * New interface `bartz.Bart` intended to supersede `bartz.BART.gbart`. `gbart` will continue to match the R package BART3, while new functionality will be added in `bartz.Bart`.
    * Built-in support for running multiple independent chains in parallel. At the interface level, this is available through `bartz.Bart` and `bartz.BART.mc_gbart`.
    * Support for multiple devices. Data and/or chains can be split across gpus. This can be controlled only through the new interface `bartz.Bart`, while `mc_gbart` will shard chains automatically on cpu but otherwise not provide settings.
    * Multivariate regression thanks to @miaoqingyu2. This is currently only implemented at low level, not in the easy to use interface.
* Performance improvements
    * Generally faster, in particular on gpu at low $n$ and on cpu with heteroskedasticity.
    * Fixed a performance regression in v0.7.0 where the running time per iteration would grow with the total number of iterations because the full trace array was duplicated on each iteration.
    * Undid the weird thing where if the number of iterations is not a multiple of `printevery` there are leftover iterations which are performed but not saved.
    * `run_mcmc` will raise an error if for any reason the MCMC code is compiled twice. This can be triggered by internal errors or by misconfiguration.
* Bugs fixed
    * Fixed wrong rare misspecified corner cases in the MCMC.
    * Fixed binary regression producing nans/infs on gpu.
    * Fixed slightly out of sync MCMC iteration logging on gpu.
* Usability improvements
    * Convenience attribute `bartz.BART.mc_gbart.sigma_` for accessing the post-burnin sigma samples.
    * Extras `bartz[cuda12]` and `bartz[cuda13]` to easily install dependencies for working with nvidia gpus; they simply mirror the corresponding jax extras.
    * Variable selection has been integrated into `bartz.mcmcstep.step`.
    * The MCMC state class `bartz.mcmcstep.State` has a new `config` attribute that, amongst other things, tracks the number of steps done on the state.


## 0.7.0 Every woman knows the pain of deciding which predictors to throw away when her design matrix is full to the brim and the last season brought out new lagged features. Our 100% money-back guarantee Bayesian variable selection Dirichlet prior will pick out the best predictors for you automatically while the MCMC is running! (2025-07-07)

The highlight of this release is the implementation of variable selection.

* Changes apparent through the `gbart` interface:
  * Parameters `sparse`, `theta`, `rho`, `a`, `b` to activate and configure variable selection
  * The MCMC logging shows a `fill` metric which is how much the tree arrays are filled, to check the trees are not being constrained by the maximum depth limit
  * Parameter `xinfo` to set manually the grid cutpoints for decision rules
  * Fixed a stochastic bug with binary regression that would become likely with >1000 datapoints
  * Parameter `rm_const` to decide how to handle "blocked" predictors that have no possible decision rules
  * The defaults of parameters `ntree` and `keepevery` now depend on whether the regression is continuous or binary, as in the R package BART
  * New attributes of `gbart` objects, matching those of R's `BART::gbart`:
    * `prob_train`, `prob_test`, `prob_train_mean`, `prob_test_mean` (for binary regression)
    * `sigma` includes burn-in samples, and `first_sigma` is gone
    * `sigma_mean` (the mean is only over kept samples)
    * `varcount`, `varcount_mean`
    * `varprob`, `varprob_mean`
* The `bartz.debug` submodule is now officially public, the main functionality is:
  * The class `debug_gbart` that adds some debugging methods to `gbart`
  * `trees_BART_to_bartz` to read trees in the format of R's BART package
  * `sample_prior` to sample from the BART prior
* Changes to internals:
  * More typing in general
  * Changes to `run_mcmc`:
    * MCMC traces are dataclasses instead of dictionaries
    * Switch back to using only one callback instead of two
      * I realized that `jax.lax.cond` makes the additional callback pointless. I previously had a cached heuristic to not use `lax.cond` because it's not efficiently vmappable, but for all practical uses in the MCMC it would be.
    * The callback accepts a jax random key, useful to implement additional MCMC steps
    * Simplified main/burn-in distinction for custom trace extractors
  * Changes to the MCMC internals:
    * `min_points_per_decision_node` intended as replacement to `min_points_per_leaf`
      * `min_points_per_leaf` is still available to allow the `gbart` interface to mimic R's BART
      * This different constraint is easier to take into account exactly in the Metropolis-Hastings ratio, while `min_points_per_leaf` leads to a deviation from the stated distribution
    * The tree structure MH should now match exactly the distributions written on paper, if `min_points_per_leaf` is not set
    * The tree structure MH never proposes zero-probability states, if `min_points_per_leaf` is not set
  * Valid usage should not produce infs or nans internally any more, so `jax.debug_infs` and `jax.debug_nans` can be used


## 0.6.0 bruv bernoulli got gauss beat any time (2025-05-29)

* binary regression with probit link
* allow to interrupt the MCMC with ^C
* logging shows how much the tree heaps are filled; if it's above 50% you should definitely increase the number of trees and/or the maximum depth
* `BART.gbart(..., run_mcmc_kw=dict(...))` allows to pass additional arguments to `mcmcloop.run_mcmc`
* option to disable logging
* refactor internals
  * set offset in the MCMC state to avoid centering the responses
  * set leaf variance in the MCMC state to avoid rescaling responses
  * immutable dataclasses instead of dicts
  * improvements to `mcmcloop.run_mcmc`
    * simpler to use signature with only three required parameters
    * the callback is allowed to carry a state and to modify the chain state (opt-in)
    * custom functions to change what is extracted from the state and put into the traces
    * two distinct callbacks, one invoked under jit and one out of it
    * more sophisticate default logging callback
  * complete type hints
  * improved documentation
  * `jaxext.split` is a less error-prone alternative to `jax.random.split`


## 0.5.0 Our work promotes diversity in error variances by following heteroscedasticity best-practices such as multiplying the variance parameter by different factors (2025-05-16)

* Heteroskedasticity with fixed weights.
* The internal MCMC functions now follow the jax convention of having the `key` parameter first in the signature.
* Fixed a bug where the MCMC callback would hang indefinitely.
* The library is now routinely tested with the least recent supported dependencies.


## 0.4.1 Told it was, nigh the end of times, version numbers of dependencies all would rise (2025-04-23)

Somehow 1 year went by before I had some time to spend on this software.

Somehow 1 year was sufficient to break most of my dependency specifications, despite having only 4 dependencies.

This release provides no new features, it's only a quick fix to make bartz go well with the latest jax and numpy versions.


## 0.4.0 The real treasure was the Markov chain samples we made along the way (2024-04-16)

* 2x faster on GPU, due to parallelizing better the tree sampling step.
* Uses less memory, now can do $n=100\,000$ with $10\,000$ trees on a V100. This was mostly an excessively large batch size for counting datapoints per leaf.
* The Metropolis-Hastings ratio is saved only for the proposed move.
* The grow and prune moves are merged into one object.


## 0.3.0

* 2-3x faster on CPU.
* Uses less memory.
* Add `initkw` argument to `BART.gbart` for advanced configuration of the MCMC initialization.
* Modified the automatic determination of `sigest` in `BART.gbart` to match the one of the R package.
* Add `usequants=False` option to `BART.gbart`, which is now the default.
* New function `prepcovars.uniform_splits_from_matrix`.
* Add `sum_trees=False` option to `grove.evaluate_forest` to evaluate separately each tree.
* Support non-batched arguments in `jaxext.autobatch`.
* Fix a bug with empty arrays in `jaxext.autobatch`.
* New option in `mcmcstep.init` to save the acceptance ratios.
* Separate batching options for residuals and counts.
* Sweeping changes to the tree move internals, more computations are parallel across trees.
* Added support for `dbarts` in the unit tests.


## 0.2.1

* Fix a bug that prevented using bart in a compiled function.


## 0.2.0

* Rename `bartz.BART` to `bartz.BART.gbart`.
* Expose submodule `bartz.jaxext` with auxiliary functions for jax.
* Shorter compilation time if no burnin or saved samples are drawn. This is useful when using the interface only to create the initial MCMC state, or when saving all samples to inspect the warm-up phase.
* 20x faster on GPU.
* 2x faster on CPU.
* Use less temporary memory to quantilize covariates, avoiding out-of-memory errors on GPU.


## 0.1.0

* Optimize the MCMC step to only traverse each tree once.
* Now `bartz` runs at the same speed as the R package `BART` (tested at $p=10$, $n=100\ldots 10000$).
* The MCMC functions are heavily changed, but the interface is the same.


## 0.0.1

* `BART` has attributes `maxdepth`, `sigest`.
* Fix errors with scaling of noise variance prior.
* Fix iteration report when `keepevery` is not 1.
* Lower required versions of dependencies to allow running on Colab.


## 0.0

First release.
