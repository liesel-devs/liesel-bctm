"""
An adapted copy of the liesel.model.distreg module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import jax
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
from liesel.goose import EngineBuilder, LieselInterface, NUTSKernel
from liesel.model import DistRegBuilder as LieselDistRegBuilder
from liesel.model import Distribution, Model, Var
from liesel.option import Option
from pandas import DataFrame

from ..custom_types import Array
from . import mi_splines as mi
from . import node as nd
from . import psplines as ps


class DistRegBuilder(LieselDistRegBuilder):
    """
    An experimental distregbuilder, providing convenience methods for
    use in Python.

    Parameters
    ----------
    data
        A dict or :class:`pd.DataFrame`. If provided, the contents of this object can
        be conveniently referenced by column name / dict key in the model-building
        methods of this class.

    """

    def __init__(self, data: dict[str, Array] | DataFrame | None = None) -> None:
        super().__init__()

        self._smooths: dict[str, list[Var]] = defaultdict(list)
        self._distributional_parameters: dict[str, Var] = {}
        self._response: Option[Var] = Option(None)
        self.data = data

    def _array(self, x: Array | str) -> Array:
        if not isinstance(x, str):
            return np.array(x, dtype=np.float32)

        if self.data is None:
            raise RuntimeError(f"No dataframe provided to find '{x}'.")

        try:
            x = self.data[x].to_numpy()
            return np.array(x, dtype=np.float32)
        except (AttributeError, TypeError):
            x = self.data[x]
            return np.array(x, dtype=np.float32)

    def add_response(
        self, y: Array | str, distribution: type[Distribution]
    ) -> DistRegBuilder:
        """Adds the response. Execute this last."""
        yval = self._array(y)
        super().add_response(yval, distribution)
        return self

    def add_param(self, name: str, inverse_link: type[tfb.Bijector]) -> DistRegBuilder:
        """
        Adds a param: a parameter of the response distribution.
        Execute this before adding the response.
        """
        super().add_predictor(name, inverse_link)
        return self

    def add_linear(
        self,
        *x: Array | str,
        m: float,
        s: float,
        param: str,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        Adds a parametric smooth to the model builder.

        Uses a Gaussian prior on the regression coefficients.

        Parameters
        ----------
        *x
            Covariates. Either arrays or strings, indicating the name of
            covariate arrays in :attr:`.data`.
        m
            The mean of the Gaussian prior.
        s
            The standard deviation of the Gaussian prior.
        param
            Name of the distributional parameter that this smooth contributes to.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        xval = np.column_stack([self._array(xvar) for xvar in x])
        str_name = self._smooth_name(name, param, "linear")

        lin = nd.Lin(str_name, xval, m, s)
        self._smooths[param].append(lin.smooth)
        self.add_groups(lin)
        return self

    def add_intercept(self, param: str) -> DistRegBuilder:
        """Adds a constant term (intercept) to the param. Has a constant prior."""
        str_name = self._smooth_name(param + "_intercept", param, "intercept")
        intercept = nd.Intercept(str_name)
        self._smooths[param].append(intercept.intercept)
        self.add_groups(intercept)
        return self

    def add_pspline(
        self,
        x: Array | str,
        nparam: int,
        a: float,
        b: float,
        param: str,
        order: int = 3,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        Adds a Bayesian P-spline smooth to the model builder.

        A sum-to-zero constraint is automatically applied to the smooth.

        Parameters
        ----------
        x
            Covariate. Either an array or a string, indicating the name
            of the covariate array in :attr:`.data`.
        nparam
            Number of parameters for the smooth.
        a
            Concentration or rate parameter of the smoothing parameter's inverse gamma
            prior.
        b
            Scale parameter of the smoothing parameter's inverse gamma prior.
        param
            Name of the distributional parameter that this smooth contributes to.
        order
            Order of the spline. Use ``order=3`` for cubic splines.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        str_name = self._smooth_name(name, param, "pspline")
        pspline = ps.PSpline(
            str_name, x=self._array(x), nparam=nparam, a=a, b=b, order=order, Z=None
        )

        self._smooths[param].append(pspline.smooth)
        self.add_groups(pspline)
        return self

    def add_teprod_full(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        param: str,
        weights: Array | None = None,
        order: int = 3,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        A tensor-product interaction based on Bayesian P-Splines.

        This term includes the marginal main effects. Uses an overall sum-to-zero
        constraint for identifiability.

        Parameters
        ----------
        x1, x2
            Covariates. Each covariate can be an array or a string that indicates the
            name of the covariate array in :attr:`.data`.
        nparam
            Tuple of integers, indicating the marginal number of parameters to use for
            each marginal smooth.
        a
            Concentration or rate parameter of the smoothing parameter's inverse gamma
            prior.
        b
            Scale parameter of the smoothing parameter's inverse gamma prior.
        param
            Name of the distributional parameter that this smooth contributes to.
        weights
            Pre-defined grid of anisotropy parameters. These are weights that determine
            which share of total variance is associated with the ``x1`` component. Must
            be larger than 0 and smaller than 1. The model will use a categorical Gibbs
            sampler to draw the best-fitting weight from this grid. If ``None``, the
            grid is initialised with ``np.linspace(0.001, 0.999, 15,
            dtype=np.float32)``.
        order
            Spline order. Use 3 (default) for cubic splines.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.

        """
        str_name = self._smooth_name(name, param, "teprod_full")
        x1val, x2val = self._array(x1), self._array(x2)

        tp_spline = ps.PSplineTP(
            str_name,
            x=(x1val, x2val),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            weights=weights,
            Z=None,
        )
        self._smooths[param].append(tp_spline.smooth)
        self.add_groups(tp_spline)
        return self

    def add_teprod_interaction(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        param: str,
        weights: Array | None = None,
        order: int = 3,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        A tensor-product interaction based on Bayesian P-Splines.

        This term excludes the marginal main effects. Uses sum-to-zero constraints on
        the marginal smooths for identifiability.

        Parameters
        ----------
        x1, x2
            Covariates. Each covariate can be an array or a string that indicates the
            name of the covariate array in :attr:`.data`.
        nparam
            Tuple of integers, indicating the marginal number of parameters to use for
            each marginal smooth.
        a
            Concentration or rate parameter of the smoothing parameter's inverse gamma
            prior.
        b
            Scale parameter of the smoothing parameter's inverse gamma prior.
        param
            Name of the distributional parameter that this smooth contributes to.
        weights
            Pre-defined grid of anisotropy parameters. These are weights that determine
            which share of total variance is associated with the ``x1`` component. Must
            be larger than 0 and smaller than 1. The model will use a categorical Gibbs
            sampler to draw the best-fitting weight from this grid. If ``None``, the
            grid is initialised with ``np.linspace(0.001, 0.999, 15,
            dtype=np.float32)``.
        order
            Spline order. Use 3 (default) for cubic splines.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        str_name = self._smooth_name(name, param, "teprod_interaction")
        x1val, x2val = self._array(x1), self._array(x2)

        tp_spline = ps.PSplineTP(
            str_name,
            x=(x1val, x2val),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            weights=weights,
            Z=(None, None),
        )
        self._smooths[param].append(tp_spline.smooth)
        self.add_groups(tp_spline)
        return self

    def add_pspline_mi(
        self,
        x: Array | str,
        nparam: int,
        a: float,
        b: float,
        param: str,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        A monotonically increasing B-spline.

        Uses the constraint :func:`.ffzero` for identifiability.

        Parameters
        ----------
        x
            Covariate. Either an array or a string that indicates the name of the
            covariate array in :attr:`.data`.
        nparam
            Tuple of integers, indicating the marginal number of parameters to use for
            each marginal smooth.
        a
            Concentration or rate parameter of the smoothing parameter's inverse gamma
            prior.
        b
            Scale parameter of the smoothing parameter's inverse gamma prior.
        param
            Name of the distributional parameter that this smooth contributes to.
        order
            Spline order. Use 3 (default) for cubic splines.
        tf
            Transformation function for the transformation from the real numbers to the
            non-negative real numbers. Should be a jax-function. Defaults to
            :func:`jax.nn.softplus`.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        str_name = self._smooth_name(name, param, "pspline_mi")
        mi_pspline = mi.MIPSpline(
            str_name,
            x=self._array(x),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            Z=None,
            positive_tranformation=positive_tranformation,
        )

        self._smooths[param].append(mi_pspline.smooth)
        self.add_groups(mi_pspline)
        return self

    def add_teprod_mi1_full(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        param: str,
        weights: Array | None = None,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        A tensor-product interaction that is monotonically increasing in the direction
        of ``x1``.

        Uses the constraint :func:`.mi_sumzero` for identifiability.

        Parameters
        ----------
        x1, x2
            Covariates. Each covariate can be an array or a string that indicates the
            name of the covariate array in :attr:`.data`.
        nparam
            Tuple of integers, indicating the marginal number of parameters to use
            for each marginal smooth.
        a
            Concentration or rate parameter of the smoothing parameter's inverse gamma
            prior.
        b
            Scale parameter of the smoothing parameter's inverse gamma prior.
        param
            Name of the distributional parameter that this smooth contributes to.
        weights
            Pre-defined grid of anisotropy parameters. These are weights that determine
            which share of total variance is associated with the ``x1`` component. Must
            be larger than 0 and smaller than 1. The model will use a categorical Gibbs
            sampler to draw the best-fitting weight from this grid. If ``None``, the
            grid is initialised with ``np.linspace(0.001, 0.999, 15,
            dtype=np.float32)``.
        order
            Spline order. Use 3 (default) for cubic splines.
        tf
            Transformation function for the transformation from the real numbers to the
            non-negative real numbers. Should be a jax-function. Defaults to
            :func:`jax.nn.softplus`.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        str_name = self._smooth_name(name, param, "teprod_mi1_full")
        x1val, x2val = self._array(x1), self._array(x2)

        mite_spline = mi.MIPSplineTE1(
            str_name,
            x=(x1val, x2val),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            weights=weights,
            positive_tranformation=positive_tranformation,
            Z=None,
        )
        self._smooths[param].append(mite_spline.smooth)
        self.add_groups(mite_spline)
        return self


def dist_reg_mcmc(model: Model, seed: int, num_chains: int) -> EngineBuilder:
    """
    Configures an :class:`.EngineBuilder` for a distributional regression model.

    The EngineBuilder uses a Metropolis-in-Gibbs MCMC algorithm with an
    :class:`.NUTSKernel` for the regression coefficients and a :class:`.GibbsKernel` for
    the smoothing parameters for a distributional regression model.

    Parameters
    ----------
    model
        A model built with a :class:`.DistRegBuilder`.
    seed
        The PRNG seed for the engine builder.
    num_chains
        The number of chains to be sampled.
    """

    builder = EngineBuilder(seed, num_chains)

    builder.set_model(LieselInterface(model))
    builder.set_initial_values(model.state)

    nuts_params = []

    for group in model.groups().values():
        for kernel in group.gibbs_kernels():  # type: ignore
            builder.add_kernel(kernel)

        nuts_params += group.sampled_params  # type: ignore

    builder.add_kernel(NUTSKernel(nuts_params))

    return builder
