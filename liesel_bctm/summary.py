from __future__ import annotations

from functools import cache, cached_property
from itertools import product
from pathlib import Path
from typing import Callable, Iterator, Sequence

import liesel.goose as gs
import numpy as np
import pandas as pd

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

        shape = np.broadcast_shapes(*[smooth.shape for smooth in smooths])
        smooths = [np.broadcast_to(smooth, shape) for smooth in smooths]

        smooth = np.sum(smooths, axis=0)
        smooth = np.moveaxis(smooth, 0, -1) if np.shape(smooth) else smooth
        return np.array(intercept + smooth)

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
            smooth_value = pt.ppeval(samples=self.samples, x=val)
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

        l = {"A": np.array([1., 2.]), "B": np.array([3., 4.]), "C": np.array([5.])}
        list(_product(**l))

    Returns::

        [
            {'A': 1.0, 'B': 3.0, 'C': 5.0},
            {'A': 1.0, 'B': 4.0, 'C': 5.0},
            {'A': 2.0, 'B': 3.0, 'C': 5.0},
            {'A': 2.0, 'B': 4.0, 'C': 5.0}
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
