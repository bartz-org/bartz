# bartz/src/bartz/testing/_dgp.py
#
# Copyright (c) 2026, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Define `gen_data` that generates simulated data for testing."""

from dataclasses import replace
from functools import partial
from typing import Literal

from equinox import Module, error_if, field
from jax import jit, random
from jax import numpy as jnp
from jax.scipy.special import hyp1f1
from jaxtyping import Array, Bool, Float, Int, Integer, Key

from bartz._jaxext import split
from bartz._jaxext.random import loggamma
from bartz.mcmcstep import OutcomeType


def generate_x(key: Key[Array, ''], n: int, p: int) -> Float[Array, 'p n']:
    """Generate predictors with mean 0 and variance 1.

    x_rj ~iid U(-√3, √3)
    """
    return random.uniform(key, (p, n), minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))


def generate_partition(key: Key[Array, ''], p: int, k: int) -> Bool[Array, 'k p']:
    """Partition x components amongst y components.

    Each row i has either p // k or p // k + 1 non-zero entries.
    """
    keys = split(key)
    indices: Int[Array, 'p'] = jnp.linspace(0, k, p, endpoint=False)
    indices = jnp.trunc(indices).astype(jnp.int32)
    indices = random.permutation(keys.pop(), indices)
    assignments: Int[Array, 'k'] = random.permutation(keys.pop(), k)
    return indices == assignments[:, None]


def generate_s(
    key: Key[Array, ''], p: int, sparsity: Float[Array, ''] | float | None
) -> Float[Array, ' p']:
    """Generate per-predictor importance scales with E[s_j^2] = 1.

    s_j ~iid Gamma(sparsity, sqrt(sparsity (sparsity + 1))), so that E[s_j^2] = 1
    and E[s_j^4] = (sparsity + 2)(sparsity + 3) / (sparsity (sparsity + 1)).
    Smaller `sparsity` spreads the importances out (sparser dependence on the
    predictors); `sparsity=None` returns all-ones scales (uniform importance).
    """
    if sparsity is None:
        return jnp.ones(p)
    else:
        log_rate = jnp.log(sparsity * (sparsity + 1)) / 2
        return jnp.exp(loggamma(key, sparsity, (p,)) - log_rate)


def generate_beta_shared(
    key: Key[Array, ''], p: int, sigma2_lin: Float[Array, ''], s: Float[Array, ' p']
) -> Float[Array, ' p']:
    """Generate shared linear coefficients for the lambda=1 case."""
    sigma2_beta = sigma2_lin / p
    return random.normal(key, (p,)) * jnp.sqrt(sigma2_beta) * s


def generate_beta_separate(
    key: Key[Array, ''],
    partition: Bool[Array, 'k p'],
    sigma2_lin: Float[Array, ''],
    s: Float[Array, ' p'],
) -> Float[Array, 'k p']:
    """Generate separate linear coefficients for the lambda=0 case."""
    k, p = partition.shape
    beta_separate: Float[Array, 'k p'] = random.normal(key, (k, p))
    sigma2_beta = sigma2_lin / (p / k)
    return jnp.where(partition, beta_separate, 0.0) * jnp.sqrt(sigma2_beta) * s


def combine_shared_separate(
    shared: Float[Array, ' n'], separate: Float[Array, 'k n'], lambda_: Float[Array, '']
) -> Float[Array, 'k n']:
    """Combine shared and separate components via the lambda_ mixing weights."""
    return jnp.sqrt(1.0 - lambda_) * separate + jnp.sqrt(lambda_) * shared


def log_var_mgf(v: Float[Array, '*shape']) -> Float[Array, '*shape']:
    R"""Log second-moment factor of one predictor's log-variance contribution.

    Returns :math:`\log E[e^{2 g X}]` for a Gaussian coefficient :math:`g \sim
    N(0, v)` and an independent predictor :math:`X \sim U(-\sqrt 3, \sqrt 3)`,
    i.e. the per-predictor factor of :math:`E[\text{error\_scale}^2]` after
    marginalizing over the coefficient. The expectation has the closed form
    :math:`\int_0^1 e^{6 v t^2}\,\mathrm dt = {}_1F_1(\tfrac12; \tfrac32; 6 v)`.
    """
    return jnp.log(hyp1f1(0.5, 1.5, 6.0 * v))


def het_normalization(
    var_coef: Float[Array, '*shape p'],
) -> tuple[Float[Array, '*shape'], Float[Array, '*shape']]:
    R"""Centering offset and dispersion of the log-linear variance multiplier.

    Given the per-predictor prior variances ``var_coef`` of the log-variance
    coefficients (one row per outcome component), returns ``(offset, var_v)``.
    ``offset`` is the constant making the multiplier :math:`v =
    \exp(2 (\sum_j g_j X_j + \text{offset}))` satisfy :math:`E_{g, X}[v] = 1`
    *marginally over the coefficient draw* (not normalized to it), so the
    marginal noise variance is preserved while the per-draw conditional variance
    fluctuates. ``var_v`` is the marginal :math:`\operatorname{Var}_{g, X}[v]`.
    Both reduce over the predictor axis; see `Params` for the closed forms.
    """
    log_mgf = log_var_mgf(var_coef)
    log_mgf_4 = log_var_mgf(4 * var_coef)
    offset = -jnp.sum(log_mgf, axis=-1) / 2
    var_v = jnp.expm1(jnp.sum(log_mgf_4 - 2 * log_mgf, axis=-1))
    return offset, var_v


def generate_het(
    key: Key[Array, ''],
    het_shape: Literal['scalar', 'vector'] | None,
    sigma2_logscale: Float[Array, ''] | float | None,
    p: int,
    partition: Bool[Array, 'k p'] | None,
    lambda_: Float[Array, ''] | float | None,
    s: Float[Array, ' p'],
) -> tuple[
    Float[Array, ' p'] | None,
    Float[Array, 'k p'] | None,
    Float[Array, ''] | Float[Array, ' k'] | None,
    Float[Array, ''] | Float[Array, ' k'] | None,
]:
    """Sample log-variance coefficients and their normalization for `gen_params`.

    Returns ``(gamma_shared, gamma_separate, het_offset, var_v)``; all ``None``
    when ``het_shape is None``. The coefficients are drawn like the linear mean
    (``gamma_shared`` first, so ``'scalar'`` and ``'vector'`` share its stream),
    while the normalization uses only their *prior variances* (so it does not
    condition on the realized draw); see `het_normalization`.
    """
    if het_shape is None:
        return None, None, None, None
    else:
        keys = split(key, 2)
        gamma_shared = generate_beta_shared(keys.pop(), p, sigma2_logscale, s)
        var_shared = sigma2_logscale / p * s**2
        if het_shape == 'vector':
            gamma_separate = generate_beta_separate(
                keys.pop(), partition, sigma2_logscale, s
            )
            k, _ = partition.shape
            var_separate = sigma2_logscale / (p / k) * s**2 * partition
            var_coef = lambda_ * var_shared + (1.0 - lambda_) * var_separate
        else:
            gamma_separate = None
            var_coef = var_shared
        het_offset, var_v = het_normalization(var_coef)
        return gamma_shared, gamma_separate, het_offset, var_v


def interaction_pattern(p: int, q: Integer[Array, ''] | int) -> Bool[Array, 'p p']:
    """Symmetric pattern where each row/col sums to q+1 (q must be even)."""
    q = error_if(q, q % 2 != 0, 'q must be even')
    q = error_if(q, q >= p, 'q must be less than p')

    i, j = jnp.ogrid[:p, :p]
    dist = jnp.minimum(jnp.abs(i - j), p - jnp.abs(i - j))
    return dist <= (q // 2)


def generate_A_shared(
    key: Key[Array, ''],
    p: int,
    q: Integer[Array, ''],
    sigma2_quad: Float[Array, ''],
    kurt_x: float,
    s: Float[Array, ' p'],
    mu4: Float[Array, ''] | float,
) -> Float[Array, 'p p']:
    """Generate shared quadratic coefficients for the lambda=1 case."""
    pattern: Bool[Array, 'p p'] = interaction_pattern(p, q)
    A_shared: Float[Array, 'p p'] = random.normal(key, (p, p))
    A_shared = jnp.where(pattern, A_shared, 0.0)
    sigma2_A = sigma2_quad / (p * ((kurt_x - 1) * mu4 + q))
    return A_shared * jnp.sqrt(sigma2_A) * s[:, None] * s[None, :]


def partitioned_interaction_pattern(
    partition: Bool[Array, 'k p'], q: Integer[Array, ''] | int
) -> Bool[Array, 'k p p']:
    """Interaction patterns over k disjoint variable sets (q even, < p // k)."""
    k, p = partition.shape
    q = error_if(q, q % 2 != 0, 'q must be even')
    q = error_if(q, q >= p // k, 'q must be less than p // k')

    indices: Int[Array, 'k p'] = jnp.cumsum(partition, axis=1)
    linear_dist: Int[Array, 'k p p'] = jnp.abs(
        indices[:, :, None] - indices[:, None, :]
    )
    num_vars: Int[Array, 'k'] = jnp.max(indices, axis=1)
    wrapped_dist: Int[Array, 'k p p'] = jnp.minimum(
        linear_dist, num_vars[:, None, None] - linear_dist
    )
    interacts: Bool[Array, 'k p p'] = wrapped_dist <= (q // 2)
    interacts = jnp.where(partition[:, :, None], interacts, False)
    return jnp.where(partition[:, None, :], interacts, False)


def generate_A_separate(
    key: Key[Array, ''],
    partition: Bool[Array, 'k p'],
    q: Integer[Array, ''],
    sigma2_quad: Float[Array, ''],
    kurt_x: float,
    s: Float[Array, ' p'],
    mu4: Float[Array, ''] | float,
) -> Float[Array, 'k p p']:
    """Generate separate quadratic coefficients for the lambda=0 case."""
    k, p = partition.shape
    A_separate: Float[Array, 'k p p'] = random.normal(key, (k, p, p))
    component_pattern: Bool[Array, 'k p p'] = partitioned_interaction_pattern(
        partition, q
    )
    A_separate = jnp.where(component_pattern, A_separate, 0.0)
    sigma2_A = sigma2_quad / (p / k * ((kurt_x - 1) * mu4 + q))
    return A_separate * jnp.sqrt(sigma2_A) * s[:, None] * s[None, :]


def generate_outcome(
    key: Key[Array, ''],
    mu: Float[Array, ' n'] | Float[Array, 'k n'],
    sigma2_eps: Float[Array, ''],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
    error_scale: Float[Array, ' n'] | Float[Array, 'k n'] | None = None,
) -> Float[Array, ' n'] | Float[Array, 'k n']:
    """Sample y from mu and sigma2_eps (see `Params` for binary semantics).

    With ``error_scale`` set, each error is multiplied by it (broadcasting a
    scalar-per-datapoint ``(n,)`` scale over all components), giving conditional
    variance ``sigma2_eps * error_scale ** 2``.
    """
    eps: Float[Array, ' n'] | Float[Array, 'k n'] = random.normal(key, mu.shape)
    scaled_eps = eps * jnp.sqrt(sigma2_eps)
    if error_scale is not None:
        scaled_eps = scaled_eps * error_scale
    latent = mu + scaled_eps
    if outcome_type is OutcomeType.continuous:
        return latent
    if outcome_type is OutcomeType.binary:
        return (latent > 0).astype(latent.dtype)
    binary_mask = jnp.array([t is OutcomeType.binary for t in outcome_type])
    return jnp.where(binary_mask[:, None], (latent > 0).astype(latent.dtype), latent)


class Params(Module):
    R"""All quantities of the data-generating process that do not depend on `n`.

    The data follows a multivariate quadratic Gaussian model. With observations
    :math:`i`, predictors :math:`j, j'` and outcome components :math:`c`, the
    predictors, coefficients and outcomes are

    .. math::
        :nowrap:

        \begin{align}
            X_{ij} &\overset{\mathrm{i.i.d.}}\sim U(-\sqrt 3, \sqrt 3),
                \quad i = 1, \ldots, n, \quad j, j' = 1, \ldots, p,
                \quad c = 1, \ldots, k, \\
            s_j &\overset{\mathrm{i.i.d.}}\sim
                \mathrm{Gamma}(\alpha,\, \sqrt{\alpha (\alpha + 1)}), \\
            E[s_j^2] &= 1, \\
            E[s_j^4] &= \mu_4 = \frac{(\alpha + 2)(\alpha + 3)}{\alpha (\alpha + 1)}, \\
            \{S_c\}_{c=1}^k &= \text{a random partition of } \{1, \ldots, p\},
                \quad \lfloor p/k \rfloor \le |S_c| \le \lceil p/k \rceil, \\
            \beta^{\mathrm{sh}}_j &\overset{\mathrm{i.i.d.}}\sim
                s_j\, N(0,\, \sigma^2_{\mathrm{lin}} / p), \\
            \beta^{\mathrm{sep}}_{cj} &\sim
                s_j\, \mathbb 1[j \in S_c]\, N(0,\, \sigma^2_{\mathrm{lin}} / (p / k)), \\
            P^{\mathrm{sh}}_{jj'} &= \begin{cases}
                    1 & \min(|j - j'|,\, p - |j - j'|) \le q / 2, \\
                    0 & \text{otherwise,}
                \end{cases}
                \quad q \bmod 2 = 0, \quad q < p, \\
            A^{\mathrm{sh}}_{jj'} &\sim s_j s_{j'}\, P^{\mathrm{sh}}_{jj'}\,
                N(0,\, \sigma^2_{\mathrm{quad}} / (p\, ((\kappa_X - 1)\mu_4 + q))), \\
            A^{\mathrm{sep}}_{cjj'} &\sim s_j s_{j'}\, P^{\mathrm{sep}}_{cjj'}\,
                N(0,\, \sigma^2_{\mathrm{quad}} / ((p / k)\, ((\kappa_X - 1)\mu_4 + q))), \\
            \mu^{\mathrm L}_{ci} &= \sqrt\lambda \textstyle\sum_j
                    \beta^{\mathrm{sh}}_j X_{ij}
                + \sqrt{1 - \lambda} \textstyle\sum_j
                    \beta^{\mathrm{sep}}_{cj} X_{ij}, \\
            \mu^{\mathrm Q}_{ci} &= \sqrt\lambda \textstyle\sum_{jj'}
                    A^{\mathrm{sh}}_{jj'} X_{ij} X_{ij'}
                + \sqrt{1 - \lambda} \textstyle\sum_{jj'}
                    A^{\mathrm{sep}}_{cjj'} X_{ij} X_{ij'},
                \quad \lambda \in [0, 1], \\
            Y_{ci} &\sim N(\mu^{\mathrm L}_{ci} + \mu^{\mathrm Q}_{ci},\,
                \sigma^2_{\mathrm{eps}}),
        \end{align}

    with binary components instead thresholded at zero, i.e. :math:`Y_{ci} =
    \mathbb 1[\mu^{\mathrm L}_{ci} + \mu^{\mathrm Q}_{ci} + \sigma_{\mathrm{eps}}
    \varepsilon_{ci} > 0]`, :math:`\varepsilon_{ci} \sim N(0, 1)` (see
    `outcome_type`). Here :math:`\kappa_X = E[X_{ij}^4] = 9/5` is the kurtosis
    of the predictors (`kurt_x`), so :math:`\kappa_X - 1 = 4/5`. The separate
    quadratic pattern :math:`P^{\mathrm{sep}}_{cjj'}` is the same circular band
    of half-width :math:`q / 2` as :math:`P^{\mathrm{sh}}`, but built on the
    within-component ranks of the predictors owned by :math:`c` and wrapped at
    :math:`|S_c|` (requiring :math:`q < \lfloor p/k \rfloor`); it is nonzero
    only for :math:`j, j' \in S_c`.

    The scales :math:`s_j` (`s`, shape :math:`\alpha` = `sparsity`) make the
    predictors differ in importance (sparsity): each down- or up-weights
    predictor :math:`j` in every term. Because they are normalized to
    :math:`E[s_j^2] = 1` the linear budget is untouched, and the only sparsity
    quantity entering the variances is :math:`\mu_4 = E[s_j^4] \ge 1` (equal to
    1 iff the :math:`s_j` are constant). They are folded into the sampled
    coefficients :math:`\beta` and :math:`A` directly, as written above. When
    ``sparsity`` is ``None`` all scales are 1 (:math:`\mu_4 = 1`), recovering
    the uniform-importance model.

    The coupling :math:`\lambda` interpolates between independent components
    (:math:`\lambda = 0`, each uses its own coefficients on its own predictors)
    and identical ones (:math:`\lambda = 1`, all share the shared
    coefficients). The per-component variance decomposition holds for every
    :math:`\lambda` (the inner :math:`\operatorname{Var}` and :math:`E` are over
    the data :math:`X` and noise at fixed coefficients, the outer over the
    coefficients):

    .. math::
        :nowrap:

        \begin{align}
            E[\operatorname{Var}[Y_{ci} \mid \beta, A]] &=
                \sigma^2_{\mathrm{lin}} + \sigma^2_{\mathrm{quad}}
                + \sigma^2_{\mathrm{eps}}
                \quad \text{(expected population variance)}, \\
            \operatorname{Var}[E[Y_{ci} \mid \beta, A]] &=
                \sigma^2_{\mathrm{quad}}\, \mu_4 / ((\kappa_X - 1)\mu_4 + q)
                \quad \text{(variance of the expected mean)}, \\
            \operatorname{Var}[Y_{ci}] &=
                E[\operatorname{Var}[Y_{ci} \mid \beta, A]]
                + \operatorname{Var}[E[Y_{ci} \mid \beta, A]]
                \quad \text{(prior variance)}.
        \end{align}

    **Heteroskedasticity.** When ``het_shape`` is not ``None`` the homoskedastic
    error :math:`\sigma_{\mathrm{eps}}\varepsilon_{ci}` is replaced by
    :math:`\sigma_{\mathrm{eps}} W_{ci}\varepsilon_{ci}`, where the positive scale
    :math:`W_{ci}` (`error_scale`) is log-linear in the predictors and shares the
    importance scales :math:`s_j` and coupling :math:`\lambda` with the mean:

    .. math::
        :nowrap:

        \begin{align}
            \gamma^{\mathrm{sh}}_j &\overset{\mathrm{i.i.d.}}\sim
                s_j\, N(0,\, \tau^2 / p), \\
            \gamma^{\mathrm{sep}}_{cj} &\sim
                s_j\, \mathbb 1[j \in S_c]\, N(0,\, \tau^2 / (p / k)), \\
            g_{cj} &= \sqrt\lambda\, \gamma^{\mathrm{sh}}_j
                + \sqrt{1 - \lambda}\, \gamma^{\mathrm{sep}}_{cj}
                \quad (\texttt{'vector'};\ \texttt{'scalar'}\text{ uses }
                g_j = \gamma^{\mathrm{sh}}_j), \\
            \nu_{cj} &= \operatorname{Var}[g_{cj}]
                = \frac{\tau^2}{p}\, s_j^2
                    \big(\lambda + (1 - \lambda)\, k\, \mathbb 1[j \in S_c]\big), \\
            L(\nu) &= E_{g \sim N(0, \nu),\, X}[e^{2 g X}]
                = \int_0^1 e^{6 \nu t^2}\,\mathrm dt
                = {}_1F_1(\tfrac12;\, \tfrac32;\, 6\nu), \\
            b_c &= -\tfrac12 \textstyle\sum_j \log L(\nu_{cj}), \\
            W_{ci} &= \exp\Big( \textstyle\sum_j g_{cj} X_{ij} + b_c \Big), \\
            Y_{ci} &\sim N\big(\mu_{ci},\, \sigma^2_{\mathrm{eps}}\, W_{ci}^2\big).
        \end{align}

    Crucially the offset :math:`b_c` uses only the coefficient *prior variances*
    :math:`\nu_{cj}`, not the realized draw, so it does not normalize in-sample:
    :math:`E_{\gamma, X}[W_{ci}^2] = 1` holds *marginally over the coefficients*
    (conditional on :math:`s`, as for :math:`\beta`), while the per-draw
    conditional :math:`E_X[W_{ci}^2 \mid \gamma]` fluctuates around 1 -- so the
    overall noise level varies from dataset to dataset. Because the multiplier
    averages to 1, every marginal variance above (``sigma2_pop``,
    ``sigma2_pri``) is unchanged; heteroskedasticity only redistributes a fixed
    noise budget. Its strength is :math:`\tau^2` = ``sigma2_logscale`` =
    :math:`E[\operatorname{Var}_X[\log W_{ci}]]`, the same for every
    :math:`\lambda` and component (like :math:`\sigma^2_{\mathrm{lin}}`). The
    marginal dispersion of the variance multiplier :math:`v_{ci} = W_{ci}^2` is

    .. math::

        \operatorname{Var}_{\gamma, X}[v_{ci}] = \exp\Big( \textstyle\sum_j
            \big( \log L(4 \nu_{cj}) - 2 \log L(\nu_{cj}) \big) \Big) - 1
        \quad (\texttt{var\_v}).

    In scalar mode (``het_shape = 'scalar'``) a single :math:`W_i`, built from
    :math:`\gamma^{\mathrm{sh}}` alone (no :math:`\lambda` or partition), scales
    the whole outcome vector (`error_scale` has shape :math:`(n,)`); in vector
    mode (``het_shape = 'vector'``, multivariate only) each component has its own
    :math:`W_{ci}` (shape :math:`(k, n)`). Binary components threshold the
    heteroskedastic latent, giving success probability :math:`\Phi(\mu_{ci} /
    (\sigma_{\mathrm{eps}} W_{ci}))`.

    For univariate outputs (``k is None``) the separate path and
    :math:`\lambda` are dropped (``partition``, ``beta_separate``,
    ``A_separate`` and ``lambda_`` are all ``None``) and :math:`\mu_i = \sum_j
    \beta^{\mathrm{sh}}_j X_{ij} + \sum_{jj'} A^{\mathrm{sh}}_{jj'} X_{ij}
    X_{ij'}`.
    """

    partition: Bool[Array, 'k p'] | None
    """Predictor-outcome assignment partition of shape (k, p), used only at
    ``lambda_ < 1``. Row ``i`` is the binary mask of predictors assigned to
    component ``i``; rows are disjoint and each has either ``p // k`` or
    ``p // k + 1`` entries. ``None`` in univariate mode (``k is None``)."""

    beta_shared: Float[Array, ' p']
    """Shared linear coefficients of shape (p,), used at ``lambda_ > 0``."""

    beta_separate: Float[Array, 'k p'] | None
    """Separate linear coefficients of shape (k, p), used at ``lambda_ < 1``.
    Row ``i`` is supported on ``partition[i]``. ``None`` in univariate
    mode (``k is None``)."""

    A_shared: Float[Array, 'p p']
    """Shared quadratic coefficients of shape (p, p), used at ``lambda_ > 0``.
    Nonzero on a symmetric band of ``q + 1`` entries per row/col."""

    A_separate: Float[Array, 'k p p'] | None
    """Separate quadratic coefficients of shape (k, p, p), used at
    ``lambda_ < 1``. Slice ``i`` is supported on the outer product of
    ``partition[i]`` with itself. ``None`` in univariate mode
    (``k is None``)."""

    s: Float[Array, ' p']
    """Per-predictor importance scales of shape (p,), with ``E[s_j ** 2] = 1``.
    Already folded into ``beta_*`` and ``A_*``; equivalent to scaling predictor
    ``j`` by ``s_j``. All ones when ``sparsity is None``."""

    mu_4_s: Float[Array, '']
    """Fourth moment ``E[s_j ** 4]`` of the importance scales, ``>= 1`` (equal
    to 1, i.e. no sparsity, when ``sparsity is None``). Sets the quadratic
    coefficient variance and ``sigma2_mean``."""

    sparsity: Float[Array, ''] | float | None
    """Gamma shape of the importance scales ``s``, or ``None`` for uniform
    importance. As ``sparsity -> inf`` the scales concentrate at 1, matching
    ``sparsity is None``; as ``sparsity -> 0`` all importance concentrates on a
    single predictor."""

    q: Integer[Array, '']
    """Number of quadratic interactions per predictor (even, ``< p // k``)."""

    lambda_: Float[Array, ''] | None
    """Coupling parameter in ``[0, 1]``. 0 is independent components,
    1 is identical components. ``None`` iff univariate (``partition is
    None``), in which case only the shared path contributes to ``mu``."""

    sigma2_lin: Float[Array, '']
    """Prior and expected population variance of the linear term of ``mu``."""

    sigma2_quad: Float[Array, '']
    """Expected population variance of the quadratic term of ``mu``."""

    sigma2_eps: Float[Array, '']
    """Variance of the additive error."""

    sigma2_mean: Float[Array, '']
    """Variance of the expected mean function."""

    sigma2_pop: Float[Array, '']
    """Expected population variance of y."""

    sigma2_pri: Float[Array, '']
    """Prior variance of y."""

    gamma_shared: Float[Array, ' p'] | None
    """Shared log-variance coefficients of shape (p,), drawn like ``beta_shared``
    with the ``sigma2_logscale`` budget. ``None`` when homoskedastic
    (``het_shape is None``)."""

    gamma_separate: Float[Array, 'k p'] | None
    """Separate log-variance coefficients of shape (k, p), used only for vector
    heteroskedasticity (``het_shape == 'vector'``). ``None`` otherwise."""

    het_offset: Float[Array, ''] | Float[Array, ' k'] | None
    """Centering of the log-variance (the ``b`` of `Params`), scalar for
    ``'scalar'`` het and shape (k,) for ``'vector'`` het, computed from the
    coefficient prior variances so that ``E[error_scale ** 2] == 1`` marginally
    over the coefficient draw. ``None`` when homoskedastic."""

    sigma2_logscale: Float[Array, ''] | None
    """Heteroskedasticity strength ``tau ** 2 = E[Var[log error_scale]]`` (the
    log-variance budget). ``None`` when homoskedastic."""

    var_v: Float[Array, ''] | Float[Array, ' k'] | None
    """Marginal dispersion ``Var[error_scale ** 2]`` of the unit-mean
    noise-variance multiplier (over the coefficient draw and predictors), scalar
    or shape (k,). ``None`` when homoskedastic."""

    outcome_type: OutcomeType | tuple[OutcomeType, ...] = field(static=True)
    """Per-component outcome type, either a single `OutcomeType` applied to
    every row, or a tuple of length ``k`` for mixed outcomes. For binary
    components the continuous latent ``mu + eps * sqrt(sigma2_eps) * error_scale``
    is thresholded at 0, yielding 0.0/1.0 floats. Unlike the standard probit
    convention used by `bartz.mcmcstep.init` (which fixes the latent noise
    variance to 1), here the binary latents share the same ``sigma2_eps``
    as the continuous ones, so the marginal success probability is
    ``Phi(mu / (sqrt(sigma2_eps) * error_scale))`` (with ``error_scale`` 1 when
    homoskedastic)."""

    het_shape: Literal['scalar', 'vector'] | None = field(default=None, static=True)
    """Heteroskedasticity mode. ``None`` is homoskedastic; ``'scalar'`` gives one
    ``error_scale`` per datapoint of shape (n,) scaling the whole outcome vector;
    ``'vector'`` (multivariate only) gives per-component scales of shape (k, n)."""

    kurt_x: float = 9 / 5  # kurtosis of uniform distribution
    """Kurtosis of the predictor distribution. Defaults to ``9 / 5``, the
    kurtosis of the uniform distribution used by `gen_data_from_params`."""


class DGP(Module):
    """Output of `gen_data` / `gen_data_from_params`: sampled data and parameters.

    See `Params` for the definition of the generative model. The ``_shared``
    fields are the ``lambda_=1`` limit (common across components), the
    ``_separate`` fields are the ``lambda_=0`` limit (independent across
    components), and the plain names are the realized mix at the sampled
    ``params.lambda_``.
    """

    x: Float[Array, 'p n']
    """Predictors of shape (p, n), marginally mean 0 and variance 1."""

    y: Float[Array, 'k n'] | Float[Array, ' n']
    """Noisy outcomes of shape (k, n), or (n,) if `gen_data` was called with
    ``k=None``."""

    mulin_shared: Float[Array, ' n']
    """Shared linear mean of shape (n,)."""

    mulin_separate: Float[Array, 'k n'] | None
    """Separate linear mean of shape (k, n), rows independent. ``None`` in
    univariate mode (``k is None``)."""

    mulin: Float[Array, 'k n'] | Float[Array, ' n']
    """Linear part of the latent mean of shape (k, n), or (n,) in univariate
    mode (``k is None``, equal to ``mulin_shared``)."""

    muquad_shared: Float[Array, ' n']
    """Shared quadratic mean of shape (n,)."""

    muquad_separate: Float[Array, 'k n'] | None
    """Separate quadratic mean of shape (k, n), rows independent. ``None`` in
    univariate mode (``k is None``)."""

    muquad: Float[Array, 'k n'] | Float[Array, ' n']
    """Quadratic part of the latent mean of shape (k, n), or (n,) in
    univariate mode (``k is None``, equal to ``muquad_shared``)."""

    mu: Float[Array, 'k n'] | Float[Array, ' n']
    """Latent mean ``mulin + muquad`` of shape (k, n), or (n,) in univariate
    mode (``k is None``)."""

    error_scale: Float[Array, ' n'] | Float[Array, 'k n'] | None
    """Per-datapoint error standard-deviation scale (the ``W`` of `Params`),
    suitable as the ``error_scale`` argument of `bartz.mcmcstep.init`. Shape
    (n,) for ``het_shape='scalar'``, (k, n) for ``'vector'``, ``None`` when
    homoskedastic."""

    params: Params
    """DGP parameters, see `Params`."""

    def split(self, n_train: int | None = None) -> tuple['DGP', 'DGP']:
        """Split the data into training and test sets.

        Parameters
        ----------
        n_train
            Number of training observations. If None, split in half.

        Returns
        -------
        Two `DGP` object with the train and test splits.
        """
        if n_train is None:
            n_train = self.x.shape[1] // 2
        assert 0 < n_train < self.x.shape[1], 'n_train must be in (0, n)'
        train = replace(
            self,
            x=self.x[:, :n_train],
            y=self.y[..., :n_train],
            mulin_shared=self.mulin_shared[:n_train],
            mulin_separate=(
                None
                if self.mulin_separate is None
                else self.mulin_separate[:, :n_train]
            ),
            mulin=self.mulin[..., :n_train],
            muquad_shared=self.muquad_shared[:n_train],
            muquad_separate=(
                None
                if self.muquad_separate is None
                else self.muquad_separate[:, :n_train]
            ),
            muquad=self.muquad[..., :n_train],
            mu=self.mu[..., :n_train],
            error_scale=(
                None if self.error_scale is None else self.error_scale[..., :n_train]
            ),
        )
        test = replace(
            self,
            x=self.x[:, n_train:],
            y=self.y[..., n_train:],
            mulin_shared=self.mulin_shared[n_train:],
            mulin_separate=(
                None
                if self.mulin_separate is None
                else self.mulin_separate[:, n_train:]
            ),
            mulin=self.mulin[..., n_train:],
            muquad_shared=self.muquad_shared[n_train:],
            muquad_separate=(
                None
                if self.muquad_separate is None
                else self.muquad_separate[:, n_train:]
            ),
            muquad=self.muquad[..., n_train:],
            mu=self.mu[..., n_train:],
            error_scale=(
                None if self.error_scale is None else self.error_scale[..., n_train:]
            ),
        )
        return train, test


@partial(jit, static_argnames=('p', 'k', 'outcome_type', 'het_shape'))
def gen_params(
    key: Key[Array, ''],
    *,
    p: int,
    k: int | None,
    q: Integer[Array, ''] | int,
    lambda_: Float[Array, ''] | float | None = None,
    sigma2_lin: Float[Array, ''] | float,
    sigma2_quad: Float[Array, ''] | float,
    sigma2_eps: Float[Array, ''] | float,
    sparsity: Float[Array, ''] | float | None = None,
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...] = 'continuous',
    sigma2_logscale: Float[Array, ''] | float | None = None,
    het_shape: Literal['scalar', 'vector'] | None = None,
) -> Params:
    """Sample DGP coefficients and parameters (no dependence on `n`).

    See `Params` for the meaning of every parameter and the generative model
    they parametrize.

    Parameters
    ----------
    key
        JAX random key.
    p
        Number of predictors.
    k
        Number of outcome components. If `None`, generate a univariate DGP
        and skip the separate code path: ``partition``, ``beta_separate``,
        ``A_separate`` and ``lambda_`` are all set to ``None`` on the returned
        `Params`, and only the shared coefficients are drawn.
    q
        See `Params`.
    lambda_
        Coupling parameter; must be ``None`` iff ``k is None``. See `Params`.
    sigma2_lin
    sigma2_quad
    sigma2_eps
        See `Params`.
    sparsity
        Shape of the Gamma distribution of the per-predictor importance scales
        ``s`` (see `Params`). Smaller values make the dependence on the
        predictors sparser; `None` (default) gives uniform importance.
    outcome_type
        ``'continuous'``, ``'binary'``, an `OutcomeType`, or a tuple of length
        ``k`` for mixed outcomes. Tuples with all elements equal are collapsed
        to the scalar form. Tuples are not allowed when ``k is None``. See
        `Params` for the semantics.
    sigma2_logscale
        Heteroskedasticity strength ``tau ** 2`` (the log-variance budget); must
        be ``None`` iff ``het_shape is None``. See `Params`.
    het_shape
        Heteroskedasticity mode: ``None`` (homoskedastic), ``'scalar'`` (one
        ``error_scale`` per datapoint, scaling the whole outcome vector), or
        ``'vector'`` (per-component scales, multivariate only). See `Params`.

    Returns
    -------
    A `Params` with the sampled coefficients and forwarded hyperparameters.

    Raises
    ------
    ValueError
        If ``outcome_type`` is a tuple whose length does not match ``k``, or
        if a tuple ``outcome_type`` is combined with ``k=None``, or if
        ``(lambda_ is None) != (k is None)``, or if
        ``(sigma2_logscale is None) != (het_shape is None)``, or if
        ``het_shape='vector'`` is combined with ``k=None``.
    """
    if (lambda_ is None) != (k is None):
        msg = (
            'lambda_ must be None when k is None'
            if k is None
            else 'lambda_ is required when k is not None'
        )
        raise ValueError(msg)

    if (sigma2_logscale is None) != (het_shape is None):
        msg = 'sigma2_logscale and het_shape must be both set or both None'
        raise ValueError(msg)
    if het_shape == 'vector' and k is None:
        msg = "het_shape='vector' requires a multivariate outcome (k != None)"
        raise ValueError(msg)

    if isinstance(outcome_type, tuple):
        if k is None:
            msg = 'tuple outcome_type requires a multivariate outcome (k != None)'
            raise ValueError(msg)
        types = tuple(OutcomeType(t) for t in outcome_type)
        if len(types) != k:
            msg = f'outcome_type has length {len(types)} but k={k}'
            raise ValueError(msg)
        outcome_type = types[0] if len(set(types)) == 1 else types
    else:
        outcome_type = OutcomeType(outcome_type)

    # E[s^4]; the scales s are normalized so that E[s^2] = 1
    if sparsity is None:
        mu_4_s = jnp.asarray(1.0)
    else:
        mu_4_s = (sparsity + 2) * (sparsity + 3) / (sparsity * (sparsity + 1))

    # claim one key per group up front, always in the same order (shared,
    # separate, heteroskedasticity, scales), so each group's stream is unchanged
    # whether or not the others are active; the splits below sit next to their use
    keys = split(key, 4)
    shared_key = keys.pop()
    separate_key = keys.pop()
    het_key = keys.pop()
    s = generate_s(keys.pop(), p, sparsity)

    shared_keys = split(shared_key)
    beta_shared = generate_beta_shared(shared_keys.pop(), p, sigma2_lin, s)
    A_shared = generate_A_shared(
        shared_keys.pop(), p, q, sigma2_quad, Params.kurt_x, s, mu_4_s
    )

    if k is None:
        partition = None
        beta_separate = None
        A_separate = None
    else:
        assert p >= k, 'p must be at least k'
        separate_keys = split(separate_key, 3)
        partition = generate_partition(separate_keys.pop(), p, k)
        beta_separate = generate_beta_separate(
            separate_keys.pop(), partition, sigma2_lin, s
        )
        A_separate = generate_A_separate(
            separate_keys.pop(), partition, q, sigma2_quad, Params.kurt_x, s, mu_4_s
        )

    gamma_shared, gamma_separate, het_offset, var_v = generate_het(
        het_key, het_shape, sigma2_logscale, p, partition, lambda_, s
    )

    # derived variances (see `Params`); cheap scalars materialized eagerly
    sigma2_mean = sigma2_quad * mu_4_s / ((Params.kurt_x - 1) * mu_4_s + q)
    sigma2_pop = sigma2_lin + sigma2_quad + sigma2_eps
    sigma2_pri = sigma2_pop + sigma2_mean

    return Params(
        partition=partition,
        beta_shared=beta_shared,
        beta_separate=beta_separate,
        A_shared=A_shared,
        A_separate=A_separate,
        s=s,
        mu_4_s=mu_4_s,
        sparsity=sparsity,
        q=q,
        lambda_=lambda_,
        sigma2_lin=sigma2_lin,
        sigma2_quad=sigma2_quad,
        sigma2_eps=sigma2_eps,
        sigma2_mean=sigma2_mean,
        sigma2_pop=sigma2_pop,
        sigma2_pri=sigma2_pri,
        gamma_shared=gamma_shared,
        gamma_separate=gamma_separate,
        het_offset=het_offset,
        sigma2_logscale=sigma2_logscale,
        var_v=var_v,
        outcome_type=outcome_type,
        het_shape=het_shape,
    )


@partial(jit, static_argnames=('n',))
def gen_data_from_params(key: Key[Array, ''], params: Params, *, n: int) -> DGP:
    """Sample predictors and outcomes given fixed `params`.

    The output ``y`` always has shape ``(k, n)``; squeezing to ``(n,)`` for
    univariate outputs is done by `gen_data`.

    Parameters
    ----------
    key
        JAX random key.
    params
        DGP parameters from `gen_params`.
    n
        Number of observations.

    Returns
    -------
    A `DGP` object with `params` and the sampled data.
    """
    keys = split(key, 2)
    p = params.beta_shared.shape[0]

    x = generate_x(keys.pop(), n, p)
    mulin_shared = params.beta_shared @ x
    muquad_shared = jnp.einsum('rs,rj,sj->j', params.A_shared, x, x)

    if params.partition is None:
        mulin_separate = None
        muquad_separate = None
        mulin = mulin_shared
        muquad = muquad_shared
    else:
        mulin_separate = params.beta_separate @ x
        muquad_separate = jnp.einsum('irs,rj,sj->ij', params.A_separate, x, x)
        mulin = combine_shared_separate(mulin_shared, mulin_separate, params.lambda_)
        muquad = combine_shared_separate(muquad_shared, muquad_separate, params.lambda_)

    mu = mulin + muquad

    # heteroskedastic error scale W = exp(eta + offset); reuses the linear-mean
    # machinery, with the offset enforcing E[W ** 2] = 1 (marginally over gamma)
    if params.het_shape is None:
        error_scale = None
    else:
        eta = params.gamma_shared @ x
        if params.het_shape == 'vector':
            eta_separate = params.gamma_separate @ x
            eta = combine_shared_separate(eta, eta_separate, params.lambda_)
        error_scale = jnp.exp(eta + params.het_offset[..., None])

    y = generate_outcome(
        keys.pop(), mu, params.sigma2_eps, params.outcome_type, error_scale
    )

    return DGP(
        x=x,
        y=y,
        mulin_shared=mulin_shared,
        mulin_separate=mulin_separate,
        mulin=mulin,
        muquad_shared=muquad_shared,
        muquad_separate=muquad_separate,
        muquad=muquad,
        mu=mu,
        error_scale=error_scale,
        params=params,
    )


@partial(jit, static_argnames=('n', 'p', 'k', 'outcome_type', 'het_shape'))
def gen_data(
    key: Key[Array, ''],
    *,
    n: int,
    p: int,
    k: int | None = None,
    q: Integer[Array, ''] | int,
    lambda_: Float[Array, ''] | float | None = None,
    sigma2_lin: Float[Array, ''] | float,
    sigma2_quad: Float[Array, ''] | float,
    sigma2_eps: Float[Array, ''] | float,
    sparsity: Float[Array, ''] | float | None = None,
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...] = 'continuous',
    sigma2_logscale: Float[Array, ''] | float | None = None,
    het_shape: Literal['scalar', 'vector'] | None = None,
) -> DGP:
    """Generate data from a quadratic multivariate DGP.

    Thin wrapper around `gen_params` followed by `gen_data_from_params`. To
    batch across `n` (e.g. to fit memory), call `gen_params` once and then
    invoke `gen_data_from_params` per batch. See `Params` for the generative
    model and `DGP` for the returned fields.

    Parameters
    ----------
    key
        JAX random key.
    n
        Number of observations.
    p
        Number of predictors.
    k
        Number of outcome components. If `None`, produces a univariate output
        with ``y.shape == (n,)`` and skips the separate code path entirely.
    q
    lambda_
    sigma2_lin
    sigma2_quad
    sigma2_eps
    sparsity
    outcome_type
    sigma2_logscale
    het_shape
        Forwarded to `gen_params`; see `Params`.

    Returns
    -------
    A `DGP` object with the sampled data and parameters.
    """
    keys = split(key, 2)
    params = gen_params(
        keys.pop(),
        p=p,
        k=k,
        q=q,
        lambda_=lambda_,
        sigma2_lin=sigma2_lin,
        sigma2_quad=sigma2_quad,
        sigma2_eps=sigma2_eps,
        sparsity=sparsity,
        outcome_type=outcome_type,
        sigma2_logscale=sigma2_logscale,
        het_shape=het_shape,
    )
    return gen_data_from_params(keys.pop(), params, n=n)
