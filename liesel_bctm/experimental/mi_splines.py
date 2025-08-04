from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax.nn import softplus
from liesel.goose import GibbsKernel
from liesel.model import Calc, Data, Var

from ..custom_types import Array
from ..distreg import constraints, gibbs
from ..distreg import psplines as ps
from ..distreg.mi_splines import _cumsum, mi_pen
from ..distreg.node import Group
from ..tp_penalty import PenaltyGroupTPExp2


class MIPSplineTE1NoCovariate2MainEffect(Group):
    """
    A tensor-product interaction that is monotonically increasing in the direction of
    the first covariate. The main effect of the second covariate is removed.

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
    ) -> None:
        A = ps.BSplineBasis(x[0], nparam[0], order=order, name=name + "_A")
        B = ps.BSplineBasis(x[1], nparam[1], order=order, name=name + "_B")

        Z2 = constraints.ffzero(nparam[1])
        B = B.reparam(Z2)

        nparam1 = A.value.shape[-1]
        nparam2 = B.value.shape[-1]

        Z = Z if Z is not None else np.eye(nparam1 * nparam2)
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
            pen2=Z2.T @ ps.pen(nparam2 + 1) @ Z2,
            Z=self.Z,
            weights=weights,
        )
        """The penalty group."""

        self.coef = ps.SplineCoef(
            self.var_group.var_param, self.pen_group, name=name + "_coef"
        )
        """The coefficient."""

        self._positive_tranformation = positive_tranformation
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


class ExperimentalTE2(Group):
    """
    A tensor-product interaction that is monotonically increasing in the direction of
    the first covariate.

    Experimental:
    - The main effect of the second covariate is removed
    - The marginal of the covariate is centered
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
    ) -> None:
        A = ps.BSplineBasis(x[0], nparam[0], order=order, name=name + "_A")
        B = ps.BSplineBasis(x[1], nparam[1], order=order, name=name + "_B")

        Z2 = constraints.ffzero(nparam[1])
        A = A.reparam(Z2)

        nparam1 = A.value.shape[-1]
        nparam2 = B.value.shape[-1]

        Z = np.eye(nparam1 * nparam2)
        S = _cumsum(nparam1, nparam2)
        C = ps.TPBasis(A, B, name=name + "_C")
        C.reparam(S)

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
        self.pen_group = PenaltyGroupTPExp2(
            name + "_penalty",
            nparam1=nparam[0],
            nparam2=nparam[1],
            Z=self.Z,
            weights=weights,
        )
        """The penalty group."""

        self.coef = ps.SplineCoef(
            self.var_group.var_param,
            self.pen_group,  # type: ignore
            name=name + "_coef",  # type: ignore
        )
        """The coefficient."""

        self._positive_tranformation = positive_tranformation
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
