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
from jaxtyping import Array, Bool, Float, Int, Integer, Key

from bartz._jaxext import split
from bartz.mcmcstep import OutcomeType
from bartz.testing._distr import Constant, DiscreteUniform, Distr, ScaleDistr, Uniform


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


def generate_beta_shared(
    key: Key[Array, ''],
    distr: Distr,
    p: int,
    sigma2_lin: Float[Array, ''],
    s: Float[Array, ' p'],
) -> Float[Array, ' p']:
    """Generate shared linear coefficients for the lambda=1 case."""
    sigma2_beta = sigma2_lin / p
    return distr.sample(key, (p,)) * jnp.sqrt(sigma2_beta) * s


def generate_beta_separate(
    key: Key[Array, ''],
    distr: Distr,
    partition: Bool[Array, 'k p'],
    sigma2_lin: Float[Array, ''],
    s: Float[Array, ' p'],
) -> Float[Array, 'k p']:
    """Generate separate linear coefficients for the lambda=0 case."""
    k, p = partition.shape
    beta_separate: Float[Array, 'k p'] = distr.sample(key, (k, p))
    sigma2_beta = sigma2_lin / (p / k)
    return jnp.where(partition, beta_separate, 0.0) * jnp.sqrt(sigma2_beta) * s


def combine_shared_separate(
    shared: Float[Array, ' n'], separate: Float[Array, 'k n'], lambda_: Float[Array, '']
) -> Float[Array, 'k n']:
    """Combine shared and separate components via the lambda_ mixing weights."""
    return jnp.sqrt(1.0 - lambda_) * separate + jnp.sqrt(lambda_) * shared


def het_var_v(
    het_shape: Literal['scalar', 'vector'],
    het_strength: Float[Array, ''] | float,
    kurt_x: Float[Array, ''] | float,
    kurt_gamma: Float[Array, ''] | float,
    mu_4: Float[Array, ''] | float,
    p: int,
    k: int | None,
    lambda_: Float[Array, ''] | float | None,
) -> Float[Array, '']:
    R"""Marginal variance :math:`\operatorname{Var}[W^2]` of the noise multiplier.

    The fully marginal dispersion of :math:`W^2 = (1 - \rho) + \rho\, \eta^2`
    (over the scales, partition, coefficients and predictors): a fixed scalar set
    by the hyperparameters, the same for every outcome component. See `Params`
    for the closed form.
    """
    # Var[W^2] is quartic in gamma, so (unlike the beta/A budgets) the kurtosis
    # of gamma_distr enters; the literal 3's are pairing counts, family-free
    excess = (kurt_gamma * kurt_x * mu_4 - 3.0) / p
    if het_shape == 'vector':
        big_lambda = lambda_**2 + (1.0 - lambda_) ** 2 * k
        cross = 6.0 * lambda_ * (1.0 - lambda_) * (kurt_x * mu_4 - 1.0) / p
        r = p % k
        excess = (
            excess * big_lambda
            + cross
            + 3.0 * (1.0 - lambda_) ** 2 * r * (k - r) / p**2
        )
    return het_strength**2 * (2.0 + excess)


def generate_het(
    key: Key[Array, ''],
    distr: Distr,
    het_shape: Literal['scalar', 'vector'] | None,
    p: int,
    partition: Bool[Array, 'k p'] | None,
    s: Float[Array, ' p'],
) -> tuple[Float[Array, ' p'] | None, Float[Array, 'k p'] | None]:
    """Sample the noise-multiplier projection coefficients for `gen_params`.

    Returns ``(gamma_shared, gamma_separate)``, both ``None`` when ``het_shape is
    None``. Drawn like the linear-mean coefficients (``gamma_shared`` first, so
    ``'scalar'`` and ``'vector'`` share its stream) at unit budget, so
    ``E[eta ** 2] == 1`` marginally; ``gamma_separate`` is used only by
    ``'vector'``.
    """
    if het_shape is None:
        return None, None
    else:
        keys = split(key, 2)
        budget = jnp.ones(())  # unit budget makes E[eta ** 2] == 1 marginally
        gamma_shared = generate_beta_shared(keys.pop(), distr, p, budget, s)
        if het_shape == 'vector':
            gamma_separate = generate_beta_separate(
                keys.pop(), distr, partition, budget, s
            )
        else:
            gamma_separate = None
        return gamma_shared, gamma_separate


def interaction_pattern(p: int, q: Integer[Array, ''] | int) -> Bool[Array, 'p p']:
    """Symmetric pattern where each row/col sums to q+1 (q must be even)."""
    q = error_if(q, q % 2 != 0, 'q must be even')
    q = error_if(q, q >= p, 'q must be less than p')

    i, j = jnp.ogrid[:p, :p]
    dist = jnp.minimum(jnp.abs(i - j), p - jnp.abs(i - j))
    return dist <= (q // 2)


def generate_A_shared(
    key: Key[Array, ''],
    distr: Distr,
    p: int,
    q: Integer[Array, ''],
    sigma2_quad: Float[Array, ''],
    kurt_x: Float[Array, ''] | float,
    s: Float[Array, ' p'],
    mu4: Float[Array, ''] | float,
) -> Float[Array, 'p p']:
    """Generate shared quadratic coefficients for the lambda=1 case."""
    pattern: Bool[Array, 'p p'] = interaction_pattern(p, q)
    A_shared: Float[Array, 'p p'] = distr.sample(key, (p, p))
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
    distr: Distr,
    partition: Bool[Array, 'k p'],
    q: Integer[Array, ''],
    sigma2_quad: Float[Array, ''],
    kurt_x: Float[Array, ''] | float,
    s: Float[Array, ' p'],
    mu4: Float[Array, ''] | float,
) -> Float[Array, 'k p p']:
    """Generate separate quadratic coefficients for the lambda=0 case."""
    k, p = partition.shape
    A_separate: Float[Array, 'k p p'] = distr.sample(key, (k, p, p))
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
) -> tuple[
    Float[Array, ' n'] | Float[Array, 'k n'], Float[Array, ' n'] | Float[Array, 'k n']
]:
    """Sample the latent z and outcome y (see `Params` for binary semantics).

    With ``error_scale`` set, each error is multiplied by it (broadcasting a
    scalar-per-datapoint ``(n,)`` scale over all components), giving conditional
    variance ``sigma2_eps * error_scale ** 2``.
    """
    eps: Float[Array, ' n'] | Float[Array, 'k n'] = random.normal(key, mu.shape)
    scaled_eps = eps * jnp.sqrt(sigma2_eps)
    if error_scale is not None:
        scaled_eps = scaled_eps * error_scale
    z = mu + scaled_eps
    if outcome_type is OutcomeType.continuous:
        y = z
    elif outcome_type is OutcomeType.binary:
        y = (z > 0).astype(z.dtype)
    else:
        binary_mask = jnp.array([t is OutcomeType.binary for t in outcome_type])
        y = jnp.where(binary_mask[:, None], (z > 0).astype(z.dtype), z)
    return z, y


def parse_outcome_type(
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...], k: int | None
) -> OutcomeType | tuple[OutcomeType, ...]:
    """Validate the `outcome_type` argument of `gen_params` and normalize it.

    Tuples with all elements equal are collapsed to the scalar form.
    """
    if isinstance(outcome_type, tuple):
        if k is None:
            msg = 'tuple outcome_type requires a multivariate outcome (k != None)'
            raise ValueError(msg)
        types = tuple(OutcomeType(t) for t in outcome_type)
        if len(types) != k:
            msg = f'outcome_type has length {len(types)} but k={k}'
            raise ValueError(msg)
        return types[0] if len(set(types)) == 1 else types
    else:
        return OutcomeType(outcome_type)


class Params(Module):
    R"""All quantities of the data-generating process that do not depend on `n`.

    The data follows a multivariate quadratic model whose ingredients are
    drawn i.i.d. from configurable families: standardized ones (mean 0,
    variance 1, see `Distr`) for the predictors :math:`X` and the coefficient
    draws :math:`b, a, g`, and a scale family (:math:`E[s^2] = 1`, see
    `ScaleDistr`) for the per-predictor importances :math:`s`. With
    observations :math:`i = 1, \ldots, n`, predictors :math:`j, j' = 1,
    \ldots, p` and outcome components :math:`c = 1, \ldots, k`:

    .. math::
        :nowrap:

        \begin{align}
            X_{ij} &\overset{\mathrm{i.i.d.}}\sim \mathtt{x\_distr},
                \quad \kappa_X = E[X_{ij}^4], \\
            s_j &\overset{\mathrm{i.i.d.}}\sim \mathtt{s\_distr},
                \quad \mu_4 = E[s_j^4], \\
            \{S_c\}_{c=1}^k &= \text{a random partition of } \{1, \ldots, p\},
                \quad \lfloor p/k \rfloor \le |S_c| \le \lceil p/k \rceil, \\
            \beta^{\mathrm{sh}}_j &= s_j \sqrt{\sigma^2_{\mathrm{lin}} / p}\;
                b_j, \quad b_j \overset{\mathrm{i.i.d.}}\sim
                \mathtt{beta\_distr}, \\
            \beta^{\mathrm{sep}}_{cj} &= s_j\, \mathbb 1[j \in S_c]
                \sqrt{\sigma^2_{\mathrm{lin}} / (p/k)}\; b'_{cj},
                \quad b'_{cj} \overset{\mathrm{i.i.d.}}\sim
                \mathtt{beta\_distr}, \\
            P^{\mathrm{sh}}_{jj'} &= \mathbb 1[\min(|j - j'|,\, p - |j - j'|)
                \le q / 2], \quad q \bmod 2 = 0, \quad q < p, \\
            P^{\mathrm{sep}}_{cjj'} &= \text{the same circular band within each }
                S_c \text{ by rank}, \quad q < \lfloor p/k \rfloor, \\
            A^{\mathrm{sh}}_{jj'} &= s_j s_{j'}\, P^{\mathrm{sh}}_{jj'}\,
                \sigma_A\, a_{jj'},
                \quad a_{jj'} \overset{\mathrm{i.i.d.}}\sim \mathtt{A\_distr},
                \quad \sigma^2_A = \frac{\sigma^2_{\mathrm{quad}}}
                    {p\, ((\kappa_X - 1)\mu_4 + q)}, \\
            A^{\mathrm{sep}}_{cjj'} &= s_j s_{j'}\, P^{\mathrm{sep}}_{cjj'}\,
                \sigma'_A\, a'_{cjj'},
                \quad a'_{cjj'} \overset{\mathrm{i.i.d.}}\sim \mathtt{A\_distr},
                \quad \sigma'^2_A = \frac{\sigma^2_{\mathrm{quad}}}
                    {(p/k)\, ((\kappa_X - 1)\mu_4 + q)}, \\
            \mu^{\mathrm L}_{ci} &= \sqrt\lambda \textstyle\sum_j
                    \beta^{\mathrm{sh}}_j X_{ij}
                + \sqrt{1 - \lambda} \textstyle\sum_j
                    \beta^{\mathrm{sep}}_{cj} X_{ij},
                \quad \lambda \in [0, 1], \\
            \mu^{\mathrm Q}_{ci} &= \sqrt\lambda \textstyle\sum_{jj'}
                    A^{\mathrm{sh}}_{jj'} X_{ij} X_{ij'}
                + \sqrt{1 - \lambda} \textstyle\sum_{jj'}
                    A^{\mathrm{sep}}_{cjj'} X_{ij} X_{ij'}, \\
            \gamma^{\mathrm{sh}}_j &= s_j\, g_j / \sqrt p,
                \quad g_j \overset{\mathrm{i.i.d.}}\sim \mathtt{gamma\_distr},
                \quad \kappa_\gamma = E[g_j^4], \\
            \gamma^{\mathrm{sep}}_{cj} &= s_j\, \mathbb 1[j \in S_c]\,
                g'_{cj} \big/ \sqrt{p/k},
                \quad g'_{cj} \overset{\mathrm{i.i.d.}}\sim
                \mathtt{gamma\_distr}, \\
            \eta_{ci} &= \begin{cases}
                    \textstyle\sum_j \gamma^{\mathrm{sh}}_j X_{ij}
                        & W \text{ scalar (same for every } c\text{)}, \\
                    \sqrt\lambda \textstyle\sum_j \gamma^{\mathrm{sh}}_j X_{ij}
                        + \sqrt{1 - \lambda} \textstyle\sum_j
                        \gamma^{\mathrm{sep}}_{cj} X_{ij}
                        & W \text{ vector},
                \end{cases} \\
            W_{ci}^2 &= (1 - \rho) + \rho\, \eta_{ci}^2,
                \quad \rho \in [0, 1], \\
            \mu_{ci} &= \mu^{\mathrm L}_{ci} + \mu^{\mathrm Q}_{ci}, \\
            Z_{ci} &\sim N\big(\mu_{ci},\, \sigma^2_{\mathrm{eps}}\, W_{ci}^2\big), \\
            Y_{ci} &= \begin{cases}
                    Z_{ci} & c \text{ continuous}, \\
                    \mathbb 1[Z_{ci} > 0] & c \text{ binary}.
                \end{cases}
        \end{align}

    A binary component thresholds its own latent :math:`Z_{ci}` at zero, so
    its success probability is :math:`\Phi(\mu_{ci} / (\sigma_{\mathrm{eps}}
    W_{ci}))` (the latent shares :math:`\sigma^2_{\mathrm{eps}}` with the
    continuous components, unlike the unit-variance probit convention of
    `bartz.mcmcstep.init`). Predictor families with :math:`\kappa_X = 1`
    (binary predictors, `DiscreteUniform` with ``m=2``) have constant squares,
    so they require :math:`q \ge 2` to keep the quadratic budget well defined.
    Univariate outcomes are the :math:`k = 1`, :math:`\lambda = 1` special
    case with the component axis dropped, and only the scalar :math:`W` is
    available.

    Writing :math:`\theta` for all the sampled coefficients and
    :math:`\operatorname{Var}[\,\cdot \mid \theta]` for the population
    variance of one dataset (over :math:`X` and noise at fixed
    :math:`\theta`), the derived variances are, for every :math:`\lambda`:

    .. math::
        :nowrap:

        \begin{align}
            E[\operatorname{Var}[Z_{ci} \mid \theta]] &=
                \sigma^2_{\mathrm{lin}} + \sigma^2_{\mathrm{quad}}
                + \sigma^2_{\mathrm{eps}}
                \quad \text{(expected population variance)}, \\
            \operatorname{Var}[E[Z_{ci} \mid \theta]] &=
                \frac{\sigma^2_{\mathrm{quad}}\, \mu_4}{(\kappa_X - 1)\mu_4 + q}
                \quad \text{(variance of the expected mean)}, \\
            \operatorname{Var}[Z_{ci}] &=
                E[\operatorname{Var}[Z_{ci} \mid \theta]]
                + \operatorname{Var}[E[Z_{ci} \mid \theta]]
                \quad \text{(prior variance)}, \\
            E[W_{ci}^2] &= 1, \qquad
                \operatorname{Var}[W_{ci}^2] = \rho^2 (2 + e), \\
            e &= \begin{cases}
                    \dfrac{\kappa_\gamma \kappa_X \mu_4 - 3}{p}
                        & W \text{ scalar}, \\[2ex]
                    \dfrac{(\lambda^2 + (1 - \lambda)^2 k)
                            (\kappa_\gamma \kappa_X \mu_4 - 3)
                            + 6 \lambda (1 - \lambda) (\kappa_X \mu_4 - 1)}{p}
                        \\ \quad
                        + \dfrac{3 (1 - \lambda)^2\, r (k - r)}{p^2},
                        \quad r = p \bmod k
                        & W \text{ vector}.
                \end{cases}
        \end{align}

    The mathematical symbols and cases map to class attributes and `gen_data`
    settings as follows:

    .. list-table::
        :header-rows: 1

        * - Symbol / case
          - Attribute / setting
        * - :math:`X_{ij}`
          - `DGP.x`
        * - :math:`\kappa_X`
          - ``x_distr.kurtosis``
        * - :math:`s_j`
          - `s`
        * - :math:`\mu_4`
          - ``s_distr.fourth_moment``
        * - :math:`\{S_c\}`
          - `partition`
        * - :math:`b_j, b'_{cj}`
          - `beta_distr`
        * - :math:`\beta^{\mathrm{sh}}`
          - `beta_shared`
        * - :math:`\beta^{\mathrm{sep}}`
          - `beta_separate`
        * - :math:`a_{jj'}, a'_{cjj'}`
          - `A_distr`
        * - :math:`A^{\mathrm{sh}}`
          - `A_shared`
        * - :math:`A^{\mathrm{sep}}`
          - `A_separate`
        * - :math:`g_j, g'_{cj}`
          - `gamma_distr`
        * - :math:`\kappa_\gamma`
          - ``gamma_distr.kurtosis``
        * - :math:`\gamma^{\mathrm{sh}}`
          - `gamma_shared`
        * - :math:`\gamma^{\mathrm{sep}}`
          - `gamma_separate`
        * - :math:`q`
          - `q`
        * - :math:`\lambda`
          - `lambda_`
        * - :math:`\sigma^2_{\mathrm{lin}}`
          - `sigma2_lin`
        * - :math:`\sigma^2_{\mathrm{quad}}`
          - `sigma2_quad`
        * - :math:`\sigma^2_{\mathrm{eps}}`
          - `sigma2_eps`
        * - :math:`\operatorname{Var}[E[Z_{ci} \mid \theta]]`
          - `sigma2_mean`
        * - :math:`E[\operatorname{Var}[Z_{ci} \mid \theta]]`
          - `sigma2_pop`
        * - :math:`\operatorname{Var}[Z_{ci}]`
          - `sigma2_pri`
        * - :math:`\rho`
          - `het_strength`
        * - :math:`\operatorname{Var}[W_{ci}^2]`
          - `var_v`
        * - :math:`W_{ci}`
          - `DGP.error_scale`
        * - :math:`W` scalar
          - ``het_shape='scalar'`` (`DGP.error_scale` of shape ``(n,)``)
        * - :math:`W` vector
          - ``het_shape='vector'``, multivariate only (`DGP.error_scale` of
            shape ``(k, n)``)
        * - :math:`W_{ci} \equiv 1` (:math:`\rho = 0`)
          - ``het_shape=None`` (the :math:`W`-related attributes are ``None``)
        * - :math:`Z_{ci}`
          - `DGP.z`
        * - :math:`Y_{ci}`
          - `DGP.y`
        * - :math:`c` continuous / binary
          - `outcome_type`
        * - univariate (:math:`k = 1`, :math:`\lambda = 1`)
          - ``k=None`` in `gen_data` (`partition`, `beta_separate`,
            `A_separate` and `lambda_` are ``None``)
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
    Already folded into ``beta_*``, ``A_*`` and ``gamma_*``; equivalent to
    scaling predictor ``j`` by ``s_j``. All ones when ``s_distr`` is
    `Constant`."""

    x_distr: Distr
    """Distribution family of the predictors. Families with kurtosis 1
    (binary predictors) require ``q >= 2`` because their squares are
    constant."""

    beta_distr: Distr
    """Distribution family of the linear coefficient draws ``b``."""

    A_distr: Distr
    """Distribution family of the quadratic coefficient draws ``a``."""

    gamma_distr: Distr
    """Distribution family of the noise projection draws ``g``. Its kurtosis
    enters ``var_v``."""

    s_distr: ScaleDistr
    """Scale family of the importance scales ``s``. More dispersed scales make
    the dependence on the predictors sparser; `Constant` is uniform importance
    (``s_j = 1``)."""

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
    """Expected population variance of the latent z."""

    sigma2_pri: Float[Array, '']
    """Prior variance of the latent z."""

    gamma_shared: Float[Array, ' p'] | None
    """Shared coefficients of shape (p,) of the latent projection ``eta``, drawn
    like ``beta_shared`` with a unit budget that cancels in the normalization.
    ``None`` when homoskedastic (``het_shape is None``)."""

    gamma_separate: Float[Array, 'k p'] | None
    """Separate projection coefficients of shape (k, p), used only for vector
    heteroskedasticity (``het_shape == 'vector'``). ``None`` otherwise."""

    het_strength: Float[Array, ''] | None
    """Heteroskedasticity knob ``rho`` in ``[0, 1]``, the fraction of the
    (expected) noise variance carried by the heteroskedastic term. 0 is
    homoskedastic (``error_scale == 1``), 1 is maximally heterogeneous.
    ``None`` when homoskedastic."""

    var_v: Float[Array, ''] | None
    """Fully marginal variance ``Var[error_scale ** 2]`` of the noise multiplier
    (see `Params` for the closed form): a fixed scalar set by the
    hyperparameters, identical for every component. ``None`` when
    homoskedastic."""

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


class DGP(Module):
    """Output of `gen_data` / `gen_data_from_params`: sampled data and parameters.

    See `Params` for the definition of the generative model. The ``_shared``
    fields are the ``lambda_=1`` limit (common across components), the
    ``_separate`` fields are the ``lambda_=0`` limit (independent across
    components), and the plain names are the realized mix at the sampled
    ``params.lambda_``.
    """

    x: Float[Array, 'p n']
    """Predictors of shape (p, n), drawn i.i.d. from the standardized family
    ``params.x_distr``."""

    y: Float[Array, 'k n'] | Float[Array, ' n']
    """Noisy outcomes of shape (k, n), or (n,) if `gen_data` was called with
    ``k=None``."""

    z: Float[Array, 'k n'] | Float[Array, ' n']
    """Latent outcomes (the ``Z`` of `Params`) of shape (k, n), or (n,) if
    `gen_data` was called with ``k=None``. `y` equals ``z`` for continuous
    components and thresholds it at 0 for binary ones."""

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
            z=self.z[..., :n_train],
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
            z=self.z[..., n_train:],
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
    x_distr: Distr = Uniform(),
    beta_distr: Distr = DiscreteUniform(2),
    A_distr: Distr = DiscreteUniform(2),
    gamma_distr: Distr = DiscreteUniform(2),
    s_distr: ScaleDistr = Constant(),
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...] = 'continuous',
    het_strength: Float[Array, ''] | float | None = None,
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
    x_distr
        Distribution family of the predictors (default `Uniform`). Binary
        predictors (`DiscreteUniform` with ``m=2``, kurtosis 1) require
        ``q >= 2`` because their squares are constant.
    beta_distr
    A_distr
        Families of the linear and quadratic coefficient draws. The default
        random signs (`DiscreteUniform` with ``m=2``) give every predictor
        exactly its share of the variance budgets, so with the default
        `Constant` scales all predictors are equally important.
    gamma_distr
        Family of the noise projection draws; its kurtosis enters ``var_v``.
    s_distr
        Scale family of the per-predictor importance scales ``s`` (e.g.
        `Gamma` or `SpikeSlab`); more dispersed scales make the dependence on
        the predictors sparser. `Constant` (default) gives uniform importance.
    outcome_type
        ``'continuous'``, ``'binary'``, an `OutcomeType`, or a tuple of length
        ``k`` for mixed outcomes. Tuples with all elements equal are collapsed
        to the scalar form. Tuples are not allowed when ``k is None``. See
        `Params` for the semantics.
    het_strength
        Heteroskedasticity knob ``rho`` in ``[0, 1]`` (0 homoskedastic, 1
        maximally heterogeneous); must be ``None`` iff ``het_shape is None``.
        See `Params`.
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
        ``(het_strength is None) != (het_shape is None)``, or if
        ``het_shape='vector'`` is combined with ``k=None``.
    """
    if (lambda_ is None) != (k is None):
        msg = (
            'lambda_ must be None when k is None'
            if k is None
            else 'lambda_ is required when k is not None'
        )
        raise ValueError(msg)

    if (het_strength is None) != (het_shape is None):
        msg = 'het_strength and het_shape must be both set or both None'
        raise ValueError(msg)
    if het_shape == 'vector' and k is None:
        msg = "het_shape='vector' requires a multivariate outcome (k != None)"
        raise ValueError(msg)

    outcome_type = parse_outcome_type(outcome_type, k)

    kurt_x = x_distr.kurtosis
    # predictors whose squares are constant (kurtosis 1, i.e. binary) leave no
    # pure-square variance, so the quadratic budget needs the interaction terms
    q = error_if(
        q, (q < 2) & (kurt_x == 1), 'q must be >= 2 when the predictors are binary'
    )

    # E[s^4]; only this moment of the scales enters the normalizers
    mu_4_s = s_distr.fourth_moment

    # claim one key per group up front, always in the same order (shared,
    # separate, heteroskedasticity, scales), so each group's stream is unchanged
    # whether or not the others are active; the splits below sit next to their use
    keys = split(key, 4)
    shared_key = keys.pop()
    separate_key = keys.pop()
    het_key = keys.pop()
    s = s_distr.sample(keys.pop(), (p,))

    shared_keys = split(shared_key)
    beta_shared = generate_beta_shared(shared_keys.pop(), beta_distr, p, sigma2_lin, s)
    A_shared = generate_A_shared(
        shared_keys.pop(), A_distr, p, q, sigma2_quad, kurt_x, s, mu_4_s
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
            separate_keys.pop(), beta_distr, partition, sigma2_lin, s
        )
        A_separate = generate_A_separate(
            separate_keys.pop(), A_distr, partition, q, sigma2_quad, kurt_x, s, mu_4_s
        )

    gamma_shared, gamma_separate = generate_het(
        het_key, gamma_distr, het_shape, p, partition, s
    )
    var_v = (
        None
        if het_shape is None
        else het_var_v(
            het_shape, het_strength, kurt_x, gamma_distr.kurtosis, mu_4_s, p, k, lambda_
        )
    )

    # derived variances (see `Params`); cheap scalars materialized eagerly
    sigma2_mean = sigma2_quad * mu_4_s / ((kurt_x - 1) * mu_4_s + q)
    sigma2_pop = sigma2_lin + sigma2_quad + sigma2_eps
    sigma2_pri = sigma2_pop + sigma2_mean

    return Params(
        partition=partition,
        beta_shared=beta_shared,
        beta_separate=beta_separate,
        A_shared=A_shared,
        A_separate=A_separate,
        s=s,
        x_distr=x_distr,
        beta_distr=beta_distr,
        A_distr=A_distr,
        gamma_distr=gamma_distr,
        s_distr=s_distr,
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
        het_strength=het_strength,
        var_v=var_v,
        outcome_type=outcome_type,
        het_shape=het_shape,
    )


@partial(jit, static_argnames=('n',))
def gen_data_from_params(key: Key[Array, ''], params: Params, *, n: int) -> DGP:
    """Sample predictors and outcomes given fixed `params`.

    The outputs ``z`` and ``y`` have shape ``(k, n)``, or ``(n,)`` if `params`
    is univariate (``params.partition is None``).

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

    x = params.x_distr.sample(keys.pop(), (p, n))
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

    # heteroskedastic error scale W = sqrt(v), v = (1 - rho) + rho eta ** 2;
    # reuses the linear-mean machinery at unit budget, so E[eta ** 2] = 1 and the
    # noise moment is controlled only marginally (E[v] = 1 over the whole prior)
    if params.het_shape is None:
        error_scale = None
    else:
        eta = params.gamma_shared @ x
        if params.het_shape == 'vector':
            eta_separate = params.gamma_separate @ x
            eta = combine_shared_separate(eta, eta_separate, params.lambda_)
        v = (1.0 - params.het_strength) + params.het_strength * eta**2
        error_scale = jnp.sqrt(v)

    z, y = generate_outcome(
        keys.pop(), mu, params.sigma2_eps, params.outcome_type, error_scale
    )

    return DGP(
        x=x,
        y=y,
        z=z,
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
    x_distr: Distr = Uniform(),
    beta_distr: Distr = DiscreteUniform(2),
    A_distr: Distr = DiscreteUniform(2),
    gamma_distr: Distr = DiscreteUniform(2),
    s_distr: ScaleDistr = Constant(),
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...] = 'continuous',
    het_strength: Float[Array, ''] | float | None = None,
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
    x_distr
    beta_distr
    A_distr
    gamma_distr
    s_distr
    outcome_type
    het_strength
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
        x_distr=x_distr,
        beta_distr=beta_distr,
        A_distr=A_distr,
        gamma_distr=gamma_distr,
        s_distr=s_distr,
        outcome_type=outcome_type,
        het_strength=het_strength,
        het_shape=het_shape,
    )
    return gen_data_from_params(keys.pop(), params, n=n)
