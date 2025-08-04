"""
Functionality for monotonically increasing B-splines.
"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax.nn import softplus
from liesel.goose import GibbsKernel
from liesel.model import Calc, Data, Var

from ..custom_types import Array
from . import constraints, gibbs
from . import psplines as ps
from .node import Group


def partial_first_diff_matrix(ncol: int) -> Array:
    return np.array(
        np.eye(ncol - 2, ncol, k=1) - np.eye(ncol - 2, ncol, k=2), dtype=np.float32
    )


def mi_pen(ncol: int):
    """Penalty matrix for shape constrained (increasing) smooth."""
    D = partial_first_diff_matrix(ncol)
    return np.array(D.T @ D, dtype=np.float32)


def _cumsum(nparam1: int, nparam2: int = 1) -> Array:
    """
    Matrix for computing the monotonically increasing spline
    coefficients as the cumulative sums of positive coefficients.

    If ``nparam2=1``, this is a lower triangular matrix of ones.
    """
    tril = np.tril(np.ones(shape=(nparam1, nparam1)))
    return np.kron(tril, np.eye(nparam2)).astype(np.float32)


def _apply_positive_tranformation(x: Array, u: int, fn: Callable[[Array], Array]):
    unconstrained = x[..., :u]
    constrained = fn(x[..., u:])
    return jnp.concatenate((unconstrained, constrained), axis=-1)


class MIPSpline(Group):
    """
    A complete univariate, monotonically increasing P-Spline.

    Parameters
    ----------
    tf
        Transformation function for the transformation from the real numbers to the
        non-negative real numbers. Should be a jax-function. Defaults to
        :func:`jnp.exp`, but another sensible choice may be :func:`jax.nn.softplus`.
    Z
        Reparameterisation matrix for ensuring identifiability. Effectively, this
        changes the sampled coefficient to ``Z.T @ coef`` with precision matrix ``Z.T @
        P @ Z``, where ``P`` is the original precision matrix. If `None`, this group
        automatically uses :func:`.ffzero`, which simply sets the first coefficient to
        zero.
    """

    return_kernels = True

    def __init__(
        self,
        name: str,
        x: Array,
        nparam: int,
        a: float,
        b: float,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = softplus,
        Z: Array | None = None,
        knot_boundaries: tuple[float, float] | None = None,
    ) -> None:
        B = ps.BSplineBasisCentered(
            x,
            nparam=nparam,
            order=order,
            name=name + "_B",
            knot_boundaries=knot_boundaries,
        )
        self.Z = Data(Z) if Z is not None else Data(constraints.ffzero(nparam))
        """Reparameterisation matrix for identifiability."""

        S = _cumsum(self.Z.value.shape[-1])

        self.X = B.reparam(self.Z.value @ S)
        """The matrix of basis function evaluations."""

        self.nparam = self.X.value.shape[-1]
        """Number of *identified* parameters associated with this Pspline."""

        self.var_group = ps.IGVariance(
            name=name + "_igvar", a=a, b=b, start_value=10000.0
        )
        """The variance parameter group."""

        # TODO: Penalty for this case
        self.pen_group = ps.PenaltyGroup(
            name + "_penalty", pen=mi_pen(nparam), Z=self.Z
        )
        """The penalty group."""

        self.coef = ps.SplineCoef(
            self.var_group.var_param, self.pen_group, name=name + "_coef"
        )
        """The sampled coefficient."""

        self._positive_tranformation = positive_tranformation
        self.positive_coef = Var(
            Calc(self._positive_tranformation, self.coef), name=name + "_positive_coef"
        )
        """
        The coefficient, transformed to the positive space.

        Since the first coefficient is fixed to zero for identifiability, all remaining
        coefficients are transformed, which nicely simplifies the process.
        """

        self.smooth = Var(Calc(jnp.dot, self.X, self.positive_coef), name=name)
        """The smooth variable."""

        self.sampled_params = [self.coef.name]
        """Names of the params that can be sampled with samplers like NUTS or IWLS."""

        contents = self.var_group.nodes_and_vars | self.pen_group.nodes_and_vars

        super().__init__(
            name,
            Z=self.Z,
            X=self.X,
            smooth=self.smooth,
            coef=self.coef,
            positive_coef=self.positive_coef,
            **contents,
        )

    def _gibbs_kernels(self) -> list[GibbsKernel]:
        var_kernel = gibbs.igvar_gibbs_kernel(self)
        return [var_kernel]

    def ppeval(self, samples: dict, x: Array | None = None) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        if self.positive_coef.name in samples:
            samples = {self.coef.name: samples[self.positive_coef.name]}
        else:
            samples = {
                self.coef.name: self._positive_tranformation(samples[self.coef.name])
            }
        return ps.ppeval(self, samples, x)


class MIPSplineTE1(Group):
    """
    A tensor-product interaction that is monotonically increasing in the direction of
    the first covariate.

    Parameters
    ----------
    tf
        Transformation function for the transformation from the real numbers to the
        non-negative real numbers. Should be a jax-function. Defaults to
        :func:`jnp.exp`, but another sensible choice may be :func:`jax.nn.softplus`.
    Z
        Reparameterisation matrix for ensuring identifiability. Effectively, this
        changes the sampled coefficient to ``Z.T @ coef`` with precision matrix ``Z.T @
        P @ Z``, where ``P`` is the original precision matrix. If `None`, this group
        automatically uses :func:`.mi_sumzero`.

    """

    return_kernels = True

    def __init__(
        self,
        name: str,
        x: tuple[Array, Array],
        nparam: tuple[int, int],
        a: float,
        b: float,
        order: int = 3,
        weights: Array | None = None,
        positive_tranformation: Callable[[Array], Array] = softplus,
        Z: Array | None = None,
        knots: tuple[Array | None, Array | None] = (None, None),
    ) -> None:
        A = ps.BSplineBasis(
            x[0], nparam[0], order=order, name=name + "_A", knots=knots[0]
        )
        B = ps.BSplineBasis(
            x[1], nparam[1], order=order, name=name + "_B", knots=knots[1]
        )

        nparam1 = A.value.shape[-1]
        nparam2 = B.value.shape[-1]

        Z = Z if Z is not None else constraints.mi_sumzero(nparam1, nparam2)
        S = _cumsum(nparam1, nparam2)
        C = ps.TPBasis(A, B, name=name + "_C")
        C.reparam(S @ Z)

        self.Z = Data(Z)
        """Reparameterisation matrix for identifiability."""

        self.X = C
        """Basis matrix / design matrix of this smooth."""

        self.nparam = self.X.value.shape[-1]
        """Number of *identified* parameters associated with this Pspline."""

        self.var_group = ps.IGVariance(
            name=name + "_igvar", a=a, b=b, start_value=10000.0
        )
        """The variance parameter group."""

        weights = (
            weights
            if weights is not None
            else np.linspace(0.001, 0.999, 15, dtype=np.float32)
        )
        self.pen_group = ps.PenaltyGroupTP(
            name + "_penalty",
            pen1=mi_pen(nparam1),
            pen2=ps.pen(nparam2),
            Z=self.Z,
            weights=weights,
        )
        """The penalty group."""

        self.coef = ps.SplineCoef(
            self.var_group.var_param, self.pen_group, name=name + "_coef"
        )
        """The coefficient."""

        self._positive_tranformation = partial(
            _apply_positive_tranformation,
            u=nparam2 - 1,
            fn=positive_tranformation,
        )
        self.positive_coef = Var(
            Calc(self._positive_tranformation, self.coef), name=name + "_positive_coef"
        )

        self.smooth = Var(Calc(jnp.dot, self.X, self.positive_coef), name=name)
        """The smooth variable."""

        self.sampled_params = [self.coef.name]
        """Names of the params that can be sampled with samplers like NUTS or IWLS."""

        contents = self.var_group.nodes_and_vars | self.pen_group.nodes_and_vars
        super().__init__(
            name,
            Z=self.Z,
            X=self.X,
            coef=self.coef,
            positive_coef=self.positive_coef,
            smooth=self.smooth,
            **contents,
        )

    def _gibbs_kernels(self) -> list[GibbsKernel]:
        var_kernel = gibbs.igvar_gibbs_kernel(self)
        weight_kernel = gibbs.weight_gibbs_kernel(self)
        return [var_kernel, weight_kernel]

    def ppeval(
        self, samples: dict, x: tuple[Array | float | None, Array | float | None]
    ) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        if self.positive_coef.name in samples:
            coef_samples = {self.coef.name: samples[self.positive_coef.name]}
        else:
            coef_samples = {
                self.coef.name: self._positive_tranformation(samples[self.coef.name])
            }
        self._positive_tranformation(samples[self.coef.name])
        return ps.ppeval(self, coef_samples, x)
