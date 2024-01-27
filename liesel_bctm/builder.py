from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.goose import EngineBuilder, NUTSKernel
from liesel.model import Calc
from liesel.model import DistRegBuilder as LieselDistRegBuilder
from liesel.model import GooseModel, Group, Model, Obs, Var
from liesel.option import Option
from pandas import DataFrame

from .custom_types import Array, TFPDistribution
from .dist import TDist
from .distreg import mi_splines as mi
from .distreg import node as nd
from .distreg import psplines as ps


class CTMBuilder(LieselDistRegBuilder):
    """
    An experimental CTMBuilder, providing convenience methods for
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

        self.pt: list[Group] = []
        """List of partial transformations."""

        self.ptd: list[Var] = []
        """List of derivatives of partial transformations with respect to the response."""

        self._response: Option[Var] = Option(None)

        self.redist = None
        """Reference distribution instance."""

        self.intercept_node = None
        """The model intercept node."""

        self.data = data
        """Model data object."""

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

    def _pt_name(self, name: str | None, prefix: str) -> str:
        """
        Generates a name for a partial transformation if the ``name`` argument is
        ``None``.
        """

        smooths = self.pt + self.ptd

        other_names = [node.name for node in smooths if node.name]

        counter = 0
        while prefix + str(counter) in other_names:
            counter += 1

        if not name:
            name = prefix + str(counter)

        if name in other_names:
            raise RuntimeError(
                f"Partial transformation {repr(name)} already exists in {repr(self)}."
            )

        if name in ["z", "zd"]:
            raise RuntimeError(f"{name} is a protected name.")

        return name

    def add_response(
        self, y: Array | str, refdist: TFPDistribution | None = None
    ) -> CTMBuilder:
        """Adds the response. Execute this last."""
        yval = self._array(y)
        refdist = refdist if refdist is not None else tfd.Normal(loc=0.0, scale=1.0)
        self.refdist = refdist

        def _sum1(*x, const):
            return const + jnp.sum(jnp.array(x), axis=0)

        def _sum2(*x):
            return jnp.sum(jnp.array(x), axis=0)

        if self.intercept_node is None:
            ct_calc = Calc(_sum2, *[pt.smooth for pt in self.pt])
        else:
            ct_calc = Calc(
                _sum1, *[pt.smooth for pt in self.pt], const=self.intercept_node
            )

        ct = Var(ct_calc, name="z")
        ctd = Var(Calc(_sum2, *self.ptd), name="zd")
        dist = TDist(ct=ct, ctd=ctd, refdist=refdist)
        y = Obs(yval, distribution=dist, name="response")

        self._response = Option(y)
        self.add(y)

        return self

    def add_linear(
        self,
        *x: Array | str,
        m: float,
        s: float,
        name: str | None = None,
    ) -> CTMBuilder:
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
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        xval = np.column_stack([self._array(xvar) for xvar in x])
        str_name = self._pt_name(name, "linear")

        lin = nd.Lin(str_name, xval, m, s)
        self.pt.append(lin)
        self.add_groups(lin)
        return self
    
    def add_linear_const(
        self,
        *x: Array | str,
        name: str | None = None,
    ) -> CTMBuilder:
        """
        Adds a parametric smooth to the model builder.


        Parameters
        ----------
        *x
            Covariates. Either arrays or strings, indicating the name of
            covariate arrays in :attr:`.data`.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        xval = np.column_stack([self._array(xvar) for xvar in x])
        str_name = self._pt_name(name, "linear_const")

        lin = nd.LinConst(str_name, xval)
        self.pt.append(lin)
        self.add_groups(lin)
        return self
    

    def add_intercept(self) -> CTMBuilder:
        """Adds a constant term (intercept). Has a constant prior."""
        str_name = self._pt_name("intercept", "intercept")
        intercept = nd.Intercept(str_name)
        self.intercept_node = intercept.intercept
        self.add_groups(intercept)
        return self

    def add_pspline(
        self,
        x: Array | str,
        nparam: int,
        a: float,
        b: float,
        order: int = 3,
        name: str | None = None,
    ) -> CTMBuilder:
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
        order
            Order of the spline. Use ``order=3`` for cubic splines.
        name
            You can give a convenient name to this smooth. Allows you to retrieve the
            smooth group object from the :meth:`.GraphBuilder.groups` and
            :meth:`.Model.groups` by an easily identiable name. The smooth name will
            also be used as prefix for all :class:`.Var`s in the smooth.
        """
        str_name = self._pt_name(name, "pspline")
        pspline = ps.PSpline(
            str_name, x=self._array(x), nparam=nparam, a=a, b=b, order=order, Z=None
        )

        self.pt.append(pspline)
        self.add_groups(pspline)
        return self

    def add_teprod_full(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        weights: Array | None = None,
        order: int = 3,
        name: str | None = None,
    ) -> CTMBuilder:
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
        str_name = self._pt_name(name, "teprod_full")
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
        self.pt.append(tp_spline)
        self.add_groups(tp_spline)
        return self

    def add_teprod_interaction(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        weights: Array | None = None,
        order: int = 3,
        name: str | None = None,
    ) -> CTMBuilder:
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
        str_name = self._pt_name(name, "teprod_interaction")
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
        self.pt.append(tp_spline)
        self.add_groups(tp_spline)
        return self

    def add_pspline_mi(
        self,
        x: Array | str,
        nparam: int,
        a: float,
        b: float,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> CTMBuilder:
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
        str_name = self._pt_name(name, "pspline_mi")
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

        self.pt.append(mi_pspline)
        self.add_groups(mi_pspline)
        return self

    def add_teprod_mi1_full(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        weights: Array | None = None,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> CTMBuilder:
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
        str_name = self._pt_name(name, "teprod_mi1_full")
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
        self.pt.append(mite_spline)
        self.add_groups(mite_spline)
        return self

    def add_trafo(
        self,
        x: Array | str,
        nparam: int,
        a: float,
        b: float,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> CTMBuilder:
        name = self._pt_name(name=name, prefix="trafo")
        self.add_pspline_mi(
            x=x,
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            positive_tranformation=positive_tranformation,
            name=name,
        )
        mips = self.groups()[name]
        mipsd = MIPSDerivative(mips, name=name + "_d")
        self.ptd.append(mipsd)
        return self

    def add_trafo_teprod_full(
        self,
        y: Array | str,
        x: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        weights: Array | None = None,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> CTMBuilder:
        """
        A tensor-product interaction that is monotonically increasing in the direction
        of ``y``.

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
        name = self._pt_name(name, "trafo_teprod_full")
        self.add_teprod_mi1_full(
            x1=y,
            x2=x,
            nparam=nparam,
            a=a,
            b=b,
            weights=weights,
            order=order,
            positive_tranformation=positive_tranformation,
            name=name,
        )

        mips = self.groups()[name]
        mipsd = MITEDerivative(mips, name=name + "_d")
        self.ptd.append(mipsd)
        return self


def ppeval_derivative(
    group,
    samples,
    x: Array | float | None | tuple[Array | float | None, Array | None],
) -> Array:
    """
    Return posterior predictive evaluation of this smooth given an array of samples.

    Uses ``group.X.value`` if ``x=None``.
    """
    if group.positive_coef.name in samples:
        samples = samples[group.positive_coef.name]
    else:
        samples = group._positive_tranformation(samples[group.coef.name])
    X = group.X.d(x)
    smooth = np.tensordot(X, samples, axes=([1], [-1]))
    return np.moveaxis(smooth, 0, -1)


class MIPSDerivative(Var):
    def __init__(
        self,
        pt: mi.MIPSpline,
        name: str = "",
    ):
        self.pt = pt
        basis_derivative = Obs(pt.X.d(), name=name + "_X")
        positive_coef = pt.positive_coef
        calc = Calc(jnp.dot, basis_derivative, positive_coef)
        super().__init__(calc, name=name)

    def ppeval(self, samples: dict, x: Array | float | None = None) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        return ppeval_derivative(self.pt, samples, x)


class MITEDerivative(Var):
    def __init__(
        self,
        pt: mi.MIPSplineTE1,
        name: str = "",
    ):
        self.pt = pt
        basis_derivative = Obs(pt.X.da(), name=name + "_X")
        positive_coef = pt.positive_coef
        calc = Calc(jnp.dot, basis_derivative, positive_coef)
        super().__init__(calc, name=name)

    def ppeval(
        self, samples: dict, x: tuple[Array | float | None, Array | float | None]
    ) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        return ppeval_derivative(self.pt, samples, x)


def ctm_mcmc(model: Model, seed: int, num_chains: int) -> EngineBuilder:
    """
    Configures an :class:`.EngineBuilder` for a conditional transformation model.

    The EngineBuilder uses a Metropolis-in-Gibbs MCMC algorithm with an
    :class:`.NUTSKernel` for the regression coefficients and a :class:`.GibbsKernel` for
    the smoothing parameters for a distributional regression model.

    Parameters
    ----------
    model
        A model built with a :class:`.CTMBuilder`.
    seed
        The PRNG seed for the engine builder.
    num_chains
        The number of chains to be sampled.
    """

    builder = EngineBuilder(seed, num_chains)

    builder.set_model(GooseModel(model))
    builder.set_initial_values(model.state)

    for group in model.groups().values():
        for kernel in group.gibbs_kernels():
            builder.add_kernel(kernel)

        if group.sampled_params:
            builder.add_kernel(NUTSKernel(group.sampled_params))

    return builder
