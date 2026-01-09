from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from functools import cache, cached_property
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import liesel.goose as gs
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

from .builder import CTMBuilder
from .custom_types import Array


def grid(*arrays) -> tuple:
    grids = np.meshgrid(*arrays)
    flattened_grids = [grid.ravel() for grid in grids]
    return tuple(flattened_grids)


class ConditionalPredictions:
    """
    Note that the returned arrays have the same axis organization as the
    :attr:`.samples` dict.

    That is: ``[chains, samples, observations]``. The ``chains`` and ``samples`` axes
    are only present if they are present in the samples.

    The keyword-arguments ``**smooths`` can be used to specify manually chosen fixed
    values for individual model terms. The pattern should be ``smooth_name=value``, but
    there are some subtleties:

    - All specified values must be of the same length or scalar.
    - For univariate smooths, you can simply specify a float or a numpy array.
    - For bivariate smooths like :class:`.MIPSplineTE1`, you have to specify a tuple
      ``(x1, x2)``. If you want to evaluate the posterior predictive conditional
      distribution for a grid of values for ``x1`` and ``x2``, you can use the helper
      function :func:`.grid` to create a tuple of arrays of the correct shape.
    - If you want to use the observed model values in any place, you can use ``None``

    Examples:

    Assume you have a :class:`.CTMBuilder` with the model terms ``"yx1"`` and ``"x2"``.
    You can obtain evaluations of the interaction term ``"yx1"`` for some fixed
    values of your choosing via::

        # samples: Posterior samples
        # ctmb: CTMBuilder instance

        y = np.linspace(0, 1, 15)
        x1 = np.linspace(0, 1, 3)

        ctmp = ConditionalPredictions(samples, ctmb, yx1=grid(y, x1))
        yx1 = ctmp.partial_ctrans()


    Assume now you want to evaluate the conditional probability density function of the
    response for the same grid of fixed values and ``"x2"`` fixed to zero::

        y = np.linspace(0, 1, 15)
        x1 = np.linspace(0, 1, 3)
        x2 = 0.0

        ctmp = ConditionalPredictions(samples, ctmb, yx1=grid(y, x1), x2=x2)
        pdf = ctmp.pdf()

    """

    def __init__(self, samples: dict, builder: CTMBuilder, **smooths):
        self.smooths = smooths
        self.samples = samples
        self.pt = builder.pt
        """Partial transformations"""
        self.ptd = builder.ptd
        """Derivatives of partial transformations with respect to the response."""
        self.refdist = builder.refdist

    @cached_property
    def smooth_names(self) -> list[str]:
        return [pt.name for pt in self.pt]

    def _validate_names(self, **kwargs) -> None:
        for name in kwargs:
            if name not in self.smooth_names:
                raise ValueError(f"{name} is not among the model's smooths.")

    @staticmethod
    def _postprocess_smooth(
        smooths: list[Array], intercept: Array | float = 0.0
    ) -> Array:
        """Handles broadcasting to correct shapes and summing up."""

        shape = jnp.broadcast_shapes(*[smooth.shape for smooth in smooths])
        smooths = [jnp.broadcast_to(smooth, shape) for smooth in smooths]

        smooth = jnp.sum(jnp.c_[smooths], axis=0)
        smooth = jnp.moveaxis(smooth, 0, -1) if jnp.shape(smooth) else smooth
        return jnp.array(intercept + smooth)

    @cache
    def intercept(self) -> Array:
        """
        Returns the squeezed intercept from the :attr:`.samples` dict.
        """
        return self.samples.get("intercept", 0.0)

    @cache
    def partial_ctrans(self) -> Array:
        """
        Evaluates specific partial transformation functions. Only the partial
        transformation functions present in :attr:`.smooths` are evaluated here.

        THis method is helpful for obtaining the values of a specific smooth.
        """
        if not self.smooths:
            raise ValueError("No smooth specified.")

        pts = [pt for pt in self.pt if pt.name in self.smooths]

        smooth_values = []

        for pt in pts:
            val = self.smooths[pt.name]
            smooth_value = pt.ppeval(samples=self.samples, x=val)  # type: ignore
            smooth_values.append(np.moveaxis(smooth_value, -1, 0))

        return self._postprocess_smooth(smooth_values)

    @cache
    def partial_ctrans_d(self):
        """
        Evaluates specific partial transformation function derivatives.
        """
        if not self.smooths:
            raise ValueError("No smooth specified.")
        smooths = self.smooths
        self._validate_names(**smooths)

        ptds = [ptd for ptd in self.ptd if ptd.pt.name in smooths]

        smooth_values = []

        for pt in ptds:
            val = smooths[pt.pt.name]
            smooth_value = pt.ppeval(samples=self.samples, x=val)
            smooth_values.append(np.moveaxis(smooth_value, -1, 0))

        return self._postprocess_smooth(smooth_values)

    @cache
    def ctrans(self):
        """Evaluates the conditional transformation function."""
        smooths = self.smooths
        self._validate_names(**smooths)

        smooth_values = []

        for pt in self.pt:
            val = smooths.get(pt.name, None)
            smooth_value = pt.ppeval(samples=self.samples, x=val)
            smooth_values.append(np.moveaxis(smooth_value, -1, 0))

        return self._postprocess_smooth(smooth_values, self.intercept())

    @cache
    def ctrans_d(self):
        """Evaluates the derivative of the conditional transformation function."""
        smooths = self.smooths
        self._validate_names(**smooths)

        smooth_values = []

        for pt in self.ptd:
            val = smooths.get(pt.pt.name, None)
            smooth_value = pt.ppeval(samples=self.samples, x=val)
            smooth_values.append(np.moveaxis(smooth_value, -1, 0))

        return self._postprocess_smooth(smooth_values)

    def log_prob(self) -> Array:
        """Evaluates the posterior predictive log probability of the response."""
        z = self.ctrans()
        z_d = self.ctrans_d()

        base_log_prob = self.refdist.log_prob(z)
        log_prob_adjustment = np.log(z_d)
        log_prob = np.add(base_log_prob, log_prob_adjustment)
        return np.array(log_prob)

    def pdf(self) -> Array:
        """
        Evaluates the posterior predictive probability density function of the response.
        """
        return np.exp(self.log_prob())

    def cdf(self) -> Array:
        """
        Evaluates the posterior predictive cumulative density function of the response.
        """
        z = self.ctrans()
        return self.refdist.cdf(z)


def approximate_inverse_(y, z, znew):
    """
    Given a grid of ``f(y) = z``, this returns an approximation of ``f^-1(znew) = ynew``
    by finding the closest grid point to each element of znew and interpolating
    linearly between the two closest grid points.
    """
    i = jnp.searchsorted(z, znew, side="right") - 1
    lo, hi = z[i], z[i + 1]
    step = hi - lo
    k = (znew - lo) / step
    k = jnp.where(jnp.isinf(k), 1.0, k)
    approx_y_new = (1.0 - k) * y[i] + (k * y[i + 1])
    return approx_y_new


def _flatten_first_two_sample_dims(samples_pytree):
    """[S, C, ...] -> [S*C, ...] for every leaf in the PyTree."""
    leaves, treedef = jax.tree_util.tree_flatten(samples_pytree)
    S, C = leaves[0].shape[:2]

    def reshape(x):
        return x.reshape((S * C,) + x.shape[2:])

    return jax.tree_util.tree_unflatten(treedef, [reshape(x) for x in leaves]), (S * C)


def ctrans_inverse(
    z: Array,
    samples: dict[str, Array],
    smooths_list: list[dict[str, Array]],
    ygrid: Array,
    builder: CTMBuilder,
) -> Array:
    """
    Compute the inverse conditional transformation for a set of covariate combinations.

    Parameters
    ----------
    z : Array
        Array of shape (N,). Target values in the transformed (z) space for which to
        compute the inverse transformation.
    samples : dict[str, Array]
        Dictionary of posterior samples. Each value is an array of shape (C, S) or (C,
        S, D), where C is the number of chains and S is the number of samples.
    smooths_list : list[dict[str, Array]]
        List of length N, where each element is a dictionary of smooth values for a
        particular covariate combination. Each array in the dictionary has shape (M, K),
        with M being the length of ygrid and K the dimension of the smooth.
    ygrid : Array
        Array of shape (M,). Grid of response values over which the transformation is
        evaluated.
    builder : CTMBuilder
        Model builder used to construct the conditional transformation.

    Returns
    -------
    Array
        Array of shape (C, S, N). For each chain, sample, and covariate combination,
        returns the inverse transformation value in the response space.

    Raises
    ------
    ValueError
        If the size of z does not match the length of smooths_list.

    Notes
    -----
    - Each element of z corresponds to one element of smooths_list.
    - The function uses JAX's vmap for efficient vectorized computation over samples and
      covariate combinations.
    """
    if z.size != len(smooths_list):
        raise ValueError(
            "Each element of z has to correspond to an element of the smooths list."
        )

    leaves, _ = jax.tree_util.tree_flatten(samples)
    C, S = leaves[0].shape[:2]

    # Elements now have shape (C*S, ...)
    samples_flat, _ = _flatten_first_two_sample_dims(samples)

    def _one_case(z, sample, **smooths):
        """
        For one z, for one sample
        """
        sample_expanded = {k: jnp.expand_dims(v, (0, 1)) for k, v in sample.items()}
        ctmp_case = ConditionalPredictions(sample_expanded, builder, **smooths)
        z_eval = ctmp_case.ctrans().squeeze()
        ynew = approximate_inverse_(ygrid, z_eval, z)
        return ynew

    # --- step 1: (prepare) stack smooths_list so we can vmap over N ---
    # smooths_stacked is a dict where each value has shape (N, M, K)
    smooths_stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *smooths_list
    )

    # Helper: for fixed z_i and smooths_i, map over the (C*S,) samples
    def _per_z(z_i, smooths_i):
        # vmap over the leading (C*S) dimension of the samples tree
        y_vec = jax.vmap(lambda sample: _one_case(z_i, sample, **smooths_i))(
            samples_flat
        )  # (C*S,)
        return y_vec.reshape(C, S)  # (C, S)

    # --- step 2: map over elements of z and smooths (length N) ---
    # in_axes=(0, 0): z over axis 0, and each leaf in smooths_stacked over axis 0
    csn = jax.vmap(_per_z, in_axes=(0, 0))(z, smooths_stacked)  # (N, C, S)

    # --- step 3: return values with shape (C, S, N) ---
    return jnp.transpose(csn, (1, 2, 0))


def ctrans_inverse2(
    z: Array,  # (C, S, N)
    samples: dict[str, Array],  # leaves have shape (C, S) or (C, S, D)
    smooths_list: list[dict[str, Array]],  # length N, each leaf (M, K)
    ygrid: Array,  # (M,)
    builder: CTMBuilder,
) -> Array:
    """
    Compute the inverse conditional transformation for a set of covariate combinations.

    Parameters
    ----------
    z : Array
        Array of shape (C, S, N). Target values in the transformed (z) space for which
        to compute the inverse transformation.
    samples : dict[str, Array]
        Dictionary of posterior samples. Each value is an array of shape (C, S) or (C,
        S, D), where C is the number of chains and S is the number of samples.
    smooths_list : list[dict[str, Array]]
        List of length N, where each element is a dictionary of smooth values for a
        particular covariate combination. Each array in the dictionary has shape (M, K),
        with M being the length of ygrid and K the dimension of the smooth.
    ygrid : Array
        Array of shape (M,). Grid of response values over which the transformation is
        evaluated.
    builder : CTMBuilder
        Model builder used to construct the conditional transformation.

    Returns
    -------
    Array
        Array of shape (C, S, N). For each chain, sample, and covariate combination,
        returns the inverse transformation value in the response space.

    Raises
    ------
    ValueError
        If the size of z (last axis) does not match the length of smooths_list.

    Notes
    -----
    - Each element of z corresponds to one element of smooths_list (last axis of z) and
      one sample.
    - The function uses JAX's vmap for efficient vectorized computation over samples and
      covariate combinations.
    """
    # --- validate shapes ---
    if len(smooths_list) == 0:
        raise ValueError("smooths_list must be non-empty.")
    if z.ndim != 3:
        raise ValueError("z must have shape (C, S, N).")
    C_z, S_z, N = z.shape

    leaves, _ = jax.tree_util.tree_flatten(samples)
    C_samp, S_samp = leaves[0].shape[:2]
    if (C_z, S_z) != (C_samp, S_samp):
        raise ValueError(
            f"z first two dims {(C_z, S_z)} must match "
            "samples first two dims {(C_samp, S_samp)}."
        )
    if N != len(smooths_list):
        raise ValueError(
            f"z third dim N={N} must equal len(smooths_list)={len(smooths_list)}."
        )

    # --- stack smooths so we can vmap over N ---
    # smooths_stacked: dict with leaves shape (N, M, K)
    smooths_stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *smooths_list
    )

    # --- flatten samples to (C*S, ...) to vmap over each (c, s) pair ---
    # samples_flat has leading dim CS = C*S
    samples_flat, _ = _flatten_first_two_sample_dims(samples)  # leaves (C*S, ...)

    CS = C_samp * S_samp
    z_flat = z.reshape(CS, N)  # (C*S, N)

    def _per_sample(sample, z_vec):
        """
        For a single (c, s) sample and its z_vec over N, compute y over N.
        """
        # Expand to fake (C, S) batch dims expected by ConditionalPredictions
        sample_expanded = {k: jnp.expand_dims(v, (0, 1)) for k, v in sample.items()}

        def _one_n(z_n, smooths_n):
            ctmp_case = ConditionalPredictions(sample_expanded, builder, **smooths_n)
            z_eval = ctmp_case.ctrans().squeeze()  # (M,)
            ynew = approximate_inverse_(ygrid, z_eval, z_n)  # scalar
            return ynew

        # Map over N: (z_n, smooths_n) -> y_n
        y_over_n = jax.vmap(_one_n, in_axes=(0, 0))(z_vec, smooths_stacked)  # (N,)
        return y_over_n

    # vmap over (C*S) samples
    ys_flat = jax.vmap(_per_sample, in_axes=(0, 0))(samples_flat, z_flat)  # (C*S, N)

    # reshape back to (C, S, N)
    return ys_flat.reshape(C_samp, S_samp, N)


def trafo_one_cquantile(
    q: float | Array,
    samples: dict[str, Array],  # leaves have shape (C, S) or (C, S, D)
    smooths_list: list[dict[str, Array]],  # length N, each leaf (M, K)
    ygrid: Array,  # (M,)
    builder: CTMBuilder,
) -> Array:
    """
    Compute the conditional quantile for a given probability level using the
    transformation model.

    Parameters
    ----------
    q : float or Array
        Quantile level(s) in (0, 1) for which to compute the conditional quantile(s).
    samples : dict[str, Array]
        Dictionary of posterior samples. Each value is an array of shape (C, S) or (C,
        S, D), where C is the number of chains and S is the number of samples.
    smooths_list : list[dict[str, Array]]
        List of length N, where each element is a dictionary of smooth values for a
        particular covariate combination. Each array in the dictionary has shape (M, K),
        with M being the length of ygrid and K the dimension of the smooth.
    ygrid : Array
        Array of shape (M,). Grid of response values over which the transformation is
        evaluated.
    builder : CTMBuilder
        Model builder used to construct the conditional transformation.

    Returns
    -------
    Array
        Array of shape (C, S, N). For each chain, sample, and covariate combination,
        returns the conditional quantile value in the response space.

    Notes
    -----
    - The quantile is computed by transforming the requested probability level `q` to
      the standard normal space and then inverting the transformation using the model.
    - Each element of the output corresponds to one covariate combination in
      `smooths_list`.
    """
    N = len(smooths_list)
    z = tfd.Normal(loc=0.0, scale=1.0).quantile(q)
    z = jnp.full((N,), fill_value=z)
    return ctrans_inverse(z, samples, smooths_list, ygrid, builder)


def trafo_cquantiles(
    q: Array,  # shape (Q,)
    samples: dict[str, Array],  # leaves have shape (C, S) or (C, S, D)
    smooths_list: list[dict[str, Array]],  # length N, each leaf (M, K)
    ygrid: Array,  # (M,)
    builder: CTMBuilder,
) -> Array:
    """
    Compute conditional quantiles for multiple probability levels using the
    transformation model.

    Parameters
    ----------
    q : Array
        Array of shape (Q,). Quantile levels in (0, 1) for which to compute the
        conditional quantiles.
    samples : dict[str, Array]
        Dictionary of posterior samples. Each value is an array of shape (C, S) or (C,
        S, D), where C is the number of chains and S is the number of samples.
    smooths_list : list[dict[str, Array]]
        List of length N, where each element is a dictionary of smooth values for a
        particular covariate combination. Each array in the dictionary has shape (M, K),
        with M being the length of ygrid and K the dimension of the smooth.
    ygrid : Array
        Array of shape (M,). Grid of response values over which the transformation is
        evaluated.
    builder : CTMBuilder
        Model builder used to construct the conditional transformation.

    Returns
    -------
    Array
        Array of shape (Q, C, S, N). For each quantile level, chain, sample, and
        covariate combination, returns the conditional quantile value in the response
        space.

    Raises
    ------
    ValueError
        If `q` does not have shape (Q,).

    Notes
    -----
    - The function iterates over each quantile level in `q` and computes the
      corresponding conditional quantile using `trafo_one_cquantile`.
    - The output is stacked along the first axis (quantile levels).
    """
    if not len(q.shape) == 1:
        raise ValueError("q must have shape (Q,)")

    results = []
    for i in range(q.shape[0]):
        ynew = trafo_one_cquantile(q[i], samples, smooths_list, ygrid, builder)
        results.append(ynew)

    ynew = jnp.c_[results]
    return ynew


def trafo_csample(
    key,
    n: int,
    samples: dict[str, Array],  # leaves have shape (C, S) or (C, S, D)
    smooths_list: list[dict[str, Array]],  # length N, each leaf (M, K)
    ygrid: Array,  # (M,)
    builder: CTMBuilder,
) -> Array:
    """
    Draw random samples from the conditional distribution using the transformation
    model.

    Parameters
    ----------
    key
        PRNG key for random number generation (JAX).
    n : int
        Number of random samples to draw.
    samples : dict[str, Array]
        Dictionary of posterior samples. Each value is an array of shape (C, S) or (C,
        S, D), where C is the number of chains and S is the number of samples.
    smooths_list : list[dict[str, Array]]
        List of length N, where each element is a dictionary of smooth values for a
        particular covariate combination. Each array in the dictionary has shape (M, K),
        with M being the length of ygrid and K the dimension of the smooth.
    ygrid : Array
        Array of shape (M,). Grid of response values over which the transformation is
        evaluated.
    builder : CTMBuilder
        Model builder used to construct the conditional transformation.

    Returns
    -------
    Array
        Array of shape (n, C, S, N). For each sample, chain, posterior draw, and
        covariate combination, returns a random draw from the conditional distribution
        in the response space.

    Notes
    -----
    - Uniform random samples are drawn and transformed to the standard normal space,
      then inverted using the model.
    - The output is stacked along the first axis (sample index).
    """
    leaves, _ = jax.tree_util.tree_flatten(samples)
    C_samp, S_samp = leaves[0].shape[:2]
    N = len(smooths_list)

    u = tfd.Uniform(low=0.0, high=1.0).sample((n, C_samp, S_samp, N), seed=key)
    z = tfd.Normal(loc=0.0, scale=1.0).quantile(u)
    results = []
    for i in range(n):
        ynew = ctrans_inverse2(z[i, ...], samples, smooths_list, ygrid, builder)
        results.append(ynew)

    ynew = jnp.c_[results]
    return ynew


def quantile_score(
    y_true: Array,
    y_pred_q: Array,
    tau: Array,
    weight_fn: Callable[[Array], Array] = lambda p: p,
) -> Array:
    """
    Compute the mean pinball loss (quantile score) per MCMC draw and quantile.

    Parameters
    ----------
    y_true : Array
        Array of shape (N,). True response values.
    y_pred_q : Array
        Array of shape (C, S, N, Q). Predicted quantiles for each chain, sample,
        observation, and quantile level.
    tau : Array
        Array of shape (Q,). Quantile levels in (0, 1).

    Returns
    -------
    Array
        Array of shape (C, S, Q). Mean pinball loss over observations for each chain,
        sample, and quantile level.

    Notes
    -----
    - The pinball loss is computed for each MCMC draw and quantile level, then averaged
      over observations.
    - The function uses JAX's vmap and lax.scan for efficient computation.
    """
    y_true = jnp.asarray(y_true)  # (N,)
    tau = jnp.asarray(tau)  # (Q,)

    C, S, N, Q = y_pred_q.shape
    y_pred_q = jnp.reshape(y_pred_q, (C * S, N, Q))

    def per_draw(pred_m: Array) -> Array:
        # pred_m: (N, Q) for one MCMC draw
        def body(carry, i):
            diff = y_true[i] - pred_m[i]  # (Q,)
            contrib = jnp.where(diff >= 0, tau * diff, (tau - 1.0) * diff)  # (Q,)
            return carry + contrib, None

        init = jnp.zeros_like(tau)
        total, _ = jax.lax.scan(body, init, jnp.arange(y_true.shape[0]))
        return total / y_true.shape[0]  # (Q,)

    quantile_score = 2 * jax.vmap(per_draw)(y_pred_q)  # (M, Q)
    quantile_score = quantile_score * weight_fn(tau)

    return quantile_score


def quantile_score_df(y_true: Array, y_pred_q: Array, tau: Array):
    """
    Compute a summary DataFrame of mean and standard deviation of quantile scores.

    Parameters
    ----------
    y_true : Array
        Array of shape (N,). True response values.
    y_pred_q : Array
        Array of shape (C, S, N, Q). Predicted quantiles for each chain, sample,
        observation, and quantile level.
    tau : Array
        Array of shape (Q,). Quantile levels in (0, 1).
    weight_fn
        A function that can be used to weigh quantiles, producing a quantile-weighted
        CRPS as described in Gneiting & Ranjan (2011), Eq. 8.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: - 'quantile_score_mean': Mean quantile score for each
        quantile level (averaged over chains and samples). - 'quantile_score_sd':
        Standard deviation of quantile score for each quantile level. - 'prob': Quantile
        levels (tau).

    Notes
    -----
    - Uses `quantile_score` to compute the pinball loss for each chain, sample, and
      quantile level.
    - Aggregates results by mean and standard deviation over chains and samples.
    """
    qs = quantile_score(y_true, y_pred_q, tau)
    mean_quantile_score = jnp.mean(qs, axis=(0))  # mean over samples

    quantile_score_std = jnp.std(qs, axis=(0))  # std over samples

    quantile_score_df = pd.DataFrame(
        {
            "quantile_score_mean": mean_quantile_score,
            "quantile_score_sd": quantile_score_std,
            "prob": tau.squeeze(),
        }
    )

    return quantile_score_df


def crps(
    y_true: Array,
    y_pred_q: Array,
    tau: Array,
    weight_fn: Callable[[Array], Array] = lambda p: p,
):
    """
    Compute the mean continuous ranked probability score (CRPS) over all samples and
    quantile levels.

    Parameters
    ----------
    y_true : Array
        Array of shape (N,). True response values.
    y_pred_q : Array
        Array of shape (C, S, N, Q). Predicted quantiles for each chain, sample,
        observation, and quantile level.
    tau : Array
        Array of shape (Q,). Quantile levels in (0, 1).
    weight_fn
        A function that can be used to weigh quantiles, producing a quantile-weighted
        CRPS as described in Gneiting & Ranjan (2011), Eq. 8.

    Returns
    -------
    float
        Mean CRPS value, averaged over all chains, samples, and quantile levels.

    Notes
    -----
    - Uses `quantile_score` to compute the pinball loss for each chain, sample, and
      quantile level.
    - Returns the mean value over all dimensions of the quantile score array.

    References
    ----------

    Gneiting, T., & Ranjan, R. (2011). Comparing Density Forecasts Using Threshold- and
    Quantile-Weighted Scoring Rules. Journal of Business & Economic Statistics, 29(3),
    411–422. https://doi.org/10.1198/jbes.2010.08110

    """
    qs = quantile_score(y_true, y_pred_q, tau, weight_fn=weight_fn)
    return qs.mean()


def summarise_samples(
    samples: dict[str, Array],
    fn: Callable,
    suffix: str = "",
    **kwargs,
) -> dict[str, Array]:
    """Apply summary function to all elements of the given samples."""
    return {key + suffix: fn(val, **kwargs) for key, val in samples.items()}


def sample_quantiles(
    samples: dict[str, Array], q: float, axis: tuple = (0, 1), suffix: str = ""
) -> dict[str, Array]:
    """Calculate quantiles of the given samples."""
    return summarise_samples(samples, fn=np.quantile, axis=axis, q=q, suffix=suffix)


def sample_means(
    samples: dict[str, Array], axis: tuple = (0, 1), suffix: str = ""
) -> dict[str, Array]:
    """Calculate mean of the given samples."""
    return summarise_samples(samples, fn=np.mean, axis=axis, suffix=suffix)


def array_to_dict(x: Array, names_prefix: str = "x") -> dict[str, Array]:
    """Turns a 2d-array into a dict."""

    if isinstance(x, float) or x.ndim == 1:
        return {f"{names_prefix}0": x}
    elif x.ndim == 2:
        return {f"{names_prefix}{i}": x[:, i] for i in range(x.shape[-1])}
    else:
        raise ValueError(f"x should have ndim <= 2, but it has x.ndim={x.ndim}")


def cdist_quantiles(
    samples: dict[str, Array],
    builder: CTMBuilder,
    q: Sequence[float] = (0.1, 0.5, 0.9),
    **smooths,
) -> pd.DataFrame:
    """
    Summarises the conditional distribution at the desired quantiles of the MCMC
    samples.
    """
    ctmp = ConditionalPredictions(samples, builder, **smooths)

    # Step 1: extract posterior predictive distribution evaluations
    data = dict()
    data["pdf"] = ctmp.pdf()
    data["cdf"] = ctmp.cdf()
    data["log_prob"] = ctmp.log_prob()

    # Step 2: aggregate posterior predictive evaluations
    agg = sample_means(data, suffix="-mean")
    for quantile in q:
        agg |= sample_quantiles(data, q=quantile, suffix=f"-q{quantile}")

    # Step 3: add covariates
    for name, smooth in smooths.items():
        if isinstance(smooth, tuple):
            x1, x2 = smooth
            agg[name + "_1"] = x1
            agg[name + "_2"] = x2
        else:
            agg |= array_to_dict(smooth, names_prefix=name)

    # Step 4: Bind together into a comfortable dataframe
    df = pd.DataFrame(agg)
    df["id"] = df.index
    wide_df = pd.wide_to_long(df, data, i="id", j="summary", sep="-", suffix=".+")
    return wide_df.reset_index()


def _expand_shape(x: Array, size: tuple[int, int]) -> Array:
    return np.concatenate(np.resize(x, size))


def cdist_rsample(
    samples: dict[str, Array],
    builder: CTMBuilder,
    size: int,
    **smooths,
) -> pd.DataFrame:
    """
    Draws a random subset of samples from the posterior predictive distribution.
    """
    ctmp = ConditionalPredictions(samples, builder, **smooths)

    # Step 1: extract posterior predictive distribution evaluations
    data = dict()
    data["pdf"] = np.concatenate(ctmp.pdf(), axis=0)
    data["cdf"] = np.concatenate(ctmp.cdf(), axis=0)
    data["log_prob"] = np.concatenate(ctmp.log_prob(), axis=0)

    n_covariate_grid = data["pdf"].shape[-1]

    all_indices = np.arange(data["pdf"].shape[0])
    selected_indices = np.random.choice(all_indices, size=size, replace=False)

    for key, value in data.items():
        data[key] = np.concatenate(value[selected_indices, :])

    for name, smooth in smooths.items():
        if isinstance(smooth, tuple):
            x1, x2 = smooth
            data[name + "_1"] = _expand_shape(x1, size=(size, n_covariate_grid))
            data[name + "_2"] = _expand_shape(x2, size=(size, n_covariate_grid))
        else:
            data[name] = _expand_shape(smooth, size=(size, n_covariate_grid))

    # Step 4: Bind together into a comfortable dataframe
    df = pd.DataFrame(data)
    df["id"] = np.repeat(np.arange(size), n_covariate_grid)
    return df


def cdist_quantiles_early(
    samples: dict[str, Array],
    builder: CTMBuilder,
    q: Sequence[float] = (0.1, 0.5, 0.9),
    **smooths,
) -> pd.DataFrame:
    """
    Summarises the conditional distribution at the desired quantiles of the MCMC
    samples.

    Uses the "late" scheme: ``X @ quantile(coef)``. CAUTION: This seems to lead to
    weird behavior.
    """
    data = {}

    for quantile in q:
        samp = sample_quantiles(samples, q=quantile)
        ctmp = ConditionalPredictions(samp, builder, **smooths)

        data[f"pdf-q{quantile}"] = ctmp.pdf()
        data[f"cdf-q{quantile}"] = ctmp.cdf()
        data[f"log_prob-q{quantile}"] = ctmp.log_prob()

    msamp = sample_means(samples)
    ctmp = ConditionalPredictions(msamp, builder, **smooths)
    data["pdf-mean"] = ctmp.pdf()
    data["cdf-mean"] = ctmp.cdf()
    data["log_prob-mean"] = ctmp.log_prob()

    for name, smooth in smooths.items():
        if isinstance(smooth, tuple):
            x1, x2 = smooth

            data[name + "_1"] = x1
            data[name + "_2"] = x2
        else:
            data[name] = smooth

    df = pd.DataFrame(data)
    df["id"] = df.index
    wide_df = pd.wide_to_long(
        df, ["pdf", "cdf", "log_prob"], i="id", j="quantile", sep="-", suffix=".+"
    )

    return wide_df.reset_index()


def partial_ctrans_quantiles(
    samples: dict[str, Array],
    builder: CTMBuilder,
    q: Sequence[float] = (0.1, 0.5, 0.9),
    **smooths,
) -> pd.DataFrame:
    """
    Evaluates the sum of partial transformation functions defined in ``builder`` and
    organizes them in a dataframe.
    """
    ctmp = ConditionalPredictions(samples, builder=builder, **smooths)

    # Step 1: extract posterior predictive smooth evaluations
    data = dict()
    data["value"] = ctmp.partial_ctrans()
    value_d = ctmp.partial_ctrans_d()
    if np.any(value_d):
        data["value_d"] = value_d

    # Step 2: aggregate posterior predictive evaluations
    agg = sample_means(data, suffix="-mean")
    for quantile in q:
        agg |= sample_quantiles(data, q=quantile, suffix=f"-q{quantile}")

    # Step 3: add covariates
    for name, smooth in smooths.items():
        if isinstance(smooth, tuple):
            x1, x2 = smooth
            agg[name + "_1"] = np.squeeze(x1)
            agg[name + "_2"] = np.squeeze(x2)
        else:
            agg[name] = np.squeeze(smooth)

    df = pd.DataFrame(agg)
    df["id"] = df.index
    wide_df = pd.wide_to_long(df, data, i="id", j="summary", sep="-", suffix=".+")
    return wide_df.reset_index()


def partial_ctrans_rsample(
    samples: dict[str, Array],
    builder: CTMBuilder,
    size: int,
    **smooths,
) -> pd.DataFrame:
    """
    Evaluates the sum of partial transformation functions defined in ``builder``, draws
    ``size`` samples from the posterior and organizes them in a dataframe.
    The column ``id`` is the sample number.
    """
    ctmp = ConditionalPredictions(samples, builder=builder, **smooths)

    # Step 1: extract posterior predictive smooth evaluations
    data = dict()
    data["value"] = np.concatenate(ctmp.partial_ctrans())
    value_d = ctmp.partial_ctrans_d()
    if np.any(value_d):
        data["value_d"] = np.concatenate(value_d)

    n_covariate_grid = data["value"].shape[-1]
    all_indices = np.arange(data["value"].shape[0])
    selected_indices = np.random.choice(all_indices, size=size, replace=False)

    for key, value in data.items():
        data[key] = np.concatenate(value[selected_indices, :])

    # Step 3: add covariates
    for name, smooth in smooths.items():
        if isinstance(smooth, tuple):
            x1, x2 = smooth
            data[name + "_1"] = _expand_shape(x1, size=(size, n_covariate_grid))
            data[name + "_2"] = _expand_shape(x2, size=(size, n_covariate_grid))
        else:
            data[name] = _expand_shape(smooth, size=(size, n_covariate_grid))

    df = pd.DataFrame(data)
    df["id"] = np.repeat(np.arange(size), n_covariate_grid)
    return df


def partial_ctrans_df_early(
    samples: dict[str, Array],
    builder: CTMBuilder,
    q: Sequence[float] = (0.1, 0.5, 0.9),
    **smooths,
) -> pd.DataFrame:
    """
    Evaluates the sum of partial transformation functions defined in ``builder`` and
    organizes them in a dataframe.

    Uses the "late" scheme: ``X @ quantile(coef)``. CAUTION: This seems to lead to
    weird behavior.
    """
    data = {}

    ctmp = ConditionalPredictions(samples, builder=builder, **smooths)

    for quantile in q:
        samp = sample_quantiles(samples, q=quantile)
        ctmp = ConditionalPredictions(samp, builder, **smooths)

        data[f"value-q{quantile}"] = ctmp.partial_ctrans()
        data[f"value_d-q{quantile}"] = ctmp.partial_ctrans_d()

    msamp = sample_means(samples)
    ctmp = ConditionalPredictions(msamp, builder, **smooths)
    data["value-mean"] = ctmp.partial_ctrans()
    data["value_d-mean"] = ctmp.partial_ctrans_d()

    for name, smooth in smooths.items():
        if isinstance(smooth, tuple):
            x1, x2 = smooth

            data[name + "_1"] = x1
            data[name + "_2"] = x2
        else:
            data[name] = smooth

    df = pd.DataFrame(data)
    df["id"] = df.index
    wide_df = pd.wide_to_long(
        df, ["value", "value_d"], i="id", j="summary", sep="-", suffix=".+"
    )
    return wide_df.reset_index()


def _product(**kwargs) -> Iterator[dict[str, float | int]]:
    """
    Creates dictionaries of all combinations of the input iterables. Example::

        l = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0]), "C": np.array([5.0])}
        list(_product(**l))

    Returns::

        [
            {"A": 1.0, "B": 3.0, "C": 5.0},
            {"A": 1.0, "B": 4.0, "C": 5.0},
            {"A": 2.0, "B": 3.0, "C": 5.0},
            {"A": 2.0, "B": 4.0, "C": 5.0},
        ]
    """
    return (dict(zip(kwargs, u)) for u in product(*kwargs.values()))


def sample_dgf(dgf: Callable[..., Array], n: int, **kwargs) -> pd.DataFrame:
    """
    Draws a grid of random samples from the data-generating function ``dgf``.

    Uses all combinations of the parameter values given in ``**kwargs``. The main use
    of this function is to draw large numbers of samples from a possibly complex
    data-generating function to compare the resulting conditional distributions with
    the posterior predictive conditional distributions obtain from a CTM.

    Parameters
    ----------
    dgf
        Data-generating function. Must take ``n`` as an argument and return a
        one-dimensional array.
    n
        Number of samples to draw for each combination of parameters in ``**kwargs``.
    **kwargs
        Parameter values to use for drawing samples.

    Returns
    -------
    A dataframe, containing a column ``y`` with the generated samples and one column
    for each parameter. Also contains a column ``index``, which is the sample index
    within each combination of parameters.
    """
    samples = []
    for kwi in _product(**kwargs):
        kwi["y"] = dgf(**kwi, n=n)
        samples.append(pd.DataFrame(kwi))
    return pd.concat(samples).reset_index()


def cache_results(
    engine_builder: gs.EngineBuilder, filename: str, use_cache: bool = True
) -> gs.engine.SamplingResults:
    fp = Path(filename)
    fp.parent.mkdir(exist_ok=True)

    if use_cache and fp.exists():
        return gs.engine.SamplingResults.pkl_load(fp)

    engine = engine_builder.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    results.pkl_save(fp)
    return results
