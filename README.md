[![PyPI](https://img.shields.io/pypi/v/bartz)](https://pypi.org/project/bartz/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13931477.svg)](https://doi.org/10.5281/zenodo.13931477)

# BART vectoriZed

An implementation of Bayesian Additive Regression Trees (BART) in JAX.

If you don't know what BART is, but know XGBoost, consider BART as a sort of Bayesian XGBoost. bartz makes BART run as fast as XGBoost.

BART is a nonparametric Bayesian regression technique. Given training predictors $X$ and responses $y$, BART finds a function to predict $y$ given $X$. The result of the inference is a sample of possible functions, representing the uncertainty over the determination of the function.

This Python module provides an implementation of BART that runs on GPU, to process large datasets faster. It is also good on CPU. Most other implementations of BART are for R, and run on CPU only.

On CPU, bartz runs at the speed of dbarts (the fastest implementation I know of) if n > 20,000, but using 1/20 of the memory. On GPU, the speed premium depends on sample size; it is convenient over CPU only for n > 10,000. The maximum speedup is currently 200x, on an Nvidia A100 and with at least 2,000,000 observations.

[This Colab notebook](https://colab.research.google.com/github/bartz-org/bartz/blob/main/docs/examples/basic_simdata.ipynb) runs bartz with n = 100,000 observations, p = 1000 predictors, 10,000 trees, for 1000 MCMC iterations, in 6 minutes.

BART is a very flexible method with many variants. This implementation provides only a small subset of the possible features. If you need a feature from [another BART implementation](https://bartz-org.github.io/bartz/docs-dev/pkglist.html) or from the BART literature, please [open an issue on github](https://github.com/bartz-org/bartz/issues).

## Links

- [Documentation (latest release)](https://bartz-org.github.io/bartz/docs)
- [Documentation (development version)](https://bartz-org.github.io/bartz/docs-dev)
- [Repository](https://github.com/bartz-org/bartz)
- [Code coverage](https://bartz-org.github.io/bartz/coverage)
- [Benchmarks](https://bartz-org.github.io/bartz/benchmarks)
- [List of BART packages](https://bartz-org.github.io/bartz/docs-dev/pkglist.html)

## Citing bartz

Article: Petrillo (2024), "Very fast Bayesian Additive Regression Trees on GPU", [arXiv:2410.23244](https://arxiv.org/abs/2410.23244).

To cite the software directly, including the specific version, use [zenodo](https://doi.org/10.5281/zenodo.13931477).
