"""
Functionality for P-Splines.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.distributions import MultivariateNormalDegenerate
from liesel.goose import GibbsKernel
from liesel.model import Calc, Data, Dist, Param, Var
from liesel_internal import splines
from liesel_internal.lookup import LookUp, LookUpCalculator

from ..custom_types import Array
from . import constraints, gibbs
from .node import Group

SplineDesign = splines.build_design_matrix_b_spline
SplineDesign_d = splines.build_design_matrix_b_spline_derivative
kn = splines.create_equidistant_knots

# ----------------------------------------------------------------------
# Variance parameter and penalty matrix for univariate splines
# ----------------------------------------------------------------------


class IGVariance(Group):
    """
    A variance parameter with a inverse gamma prior.

    Parameters
    ----------
    name
        Name of the group.
    a
        Concentration or rate parameter of the inverse gamma prior.
    b
        Scale parameter of the inverse gamma prior.
    start_value
        Initial value of the variance parameter.
    """

    def __init__(self, name: str, a: float, b: float, start_value: float = 100.0):

        self.a = Var(a, name=name + "_a")
        """Concentration/rate parameter of the inverse gamma prior."""

        self.b = Var(b, name=name + "_b")
        """Scale parameter of the inverse gamma prior."""

        prior = Dist(tfd.InverseGamma, concentration=self.a, scale=self.b)
        self.var_param = Param(start_value, prior, name)
        """The variance parameter."""

        super().__init__(name=name, var_param=self.var_param, a=self.a, b=self.b)


def pen(ncol: int, diff: int = 2):
    """A P-spline penalty matrix based on ``diff``-order differences."""
    D = np.diff(np.identity(ncol), diff, axis=0)
    return np.array(D.T @ D, dtype=np.float32)


class _RankDet:
    """
    The rank and generalized log determinant of the potentially rank-deficient
    matrix ``R``.
    """

    def __init__(self, R: Array, tol: float = 1e-6):
        eigenvals = np.linalg.eigvalsh(R)
        mask = eigenvals > tol
        selected = np.where(mask, eigenvals, 1.0)
        self.log_pdet = np.sum(np.log(selected)).astype(np.float32)
        self.rank = mask.sum(dtype=np.float32)


class PenaltyGroup(Group):
    """Holds the a penalty matrix and related data."""

    def __init__(self, name: str, pen: Array, Z: Data | None = None):
        if Z is not None:
            pen = Z.value.T @ pen @ Z.value  # apply constraint

        self.nparam = pen.shape[-1]
        """Number of *identified* parameters associated with the penalty matrix."""

        rk_log_pdet = _RankDet(pen)

        self.penalty = Var(pen, name=name)
        """The penalty matrix."""
        self.rank = Data(rk_log_pdet.rank, _name=name + "_rank")
        """The rank of the penalty matrix."""

        self.log_pdet = Data(rk_log_pdet.log_pdet, _name=name + "_log_pdet")
        """The pseudo-log-determinant of the penalty matrix."""

        super().__init__(
            name=name,
            penalty=self.penalty,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )


# ----------------------------------------------------------------------
# Penalty matrices for tensor product interactions
# ----------------------------------------------------------------------


def _assert_square(a: Array) -> int:
    """
    Verifies that the input array is a square matrix and returns its dimension.
    """
    if not len(a.shape) == 2:
        raise ValueError("Array must have exactly two dimensions.")

    nrow, ncol = a.shape
    if not nrow == ncol:
        raise ValueError("Matrix must be square.")

    return nrow


def tp_penalty(pen1: Array, pen2: Array, weight: float, augment: bool = False) -> Array:
    """
    Constructs a penalty matrix for the multivariate normal prior of a
    generalized interaction using an anisotropy parameter ``weight``.

    See Kneib et al. (2019) [#lego]_

    Parameters
    ----------
    pen1, pen2
        Prior penalty matrices.
    weight
        Anisotropy parameter; a weight that must be ``0.0 < weight <
        1.0``.
    augment
        If *True*, will add a small number to the diagonal for numerical
        stability.

    Returns
    -------
    Penalty matrix for the mulativariate normal prior of a generalized
    interaction.

    References
    ----------

    .. [#lego] https://doi.org/10.1007/s11749-019-00631-z
    """
    nparam1 = pen1.shape[-1]
    nparam2 = pen2.shape[-1]
    penalty1 = np.kron(pen1, np.eye(nparam2))
    penalty2 = np.kron(np.eye(nparam1), pen2)
    penalty = np.array(weight * penalty1 + (1.0 - weight) * penalty2, dtype=np.float32)

    if augment:
        # Add small values to diag to increase numerical stability
        # This is not EXACTLY as Carlan (2022) described in the paper, because this
        # penalty
        # matrix here gets multiplied with 1/tau2 later.
        # Carlan (2022) augment the full precision matrix, not only the penalty matrix.
        di = np.diag_indices(penalty.shape[-1])
        np.fill_diagonal(penalty, penalty[di] + 1e-5)
        # penalty = penalty.at[di].set(penalty[di] + 1e-5)

    return penalty


class _PenaltyGridTP:
    def __init__(
        self,
        pen1: Array,
        pen2: Array,
        weights: Array,
        Z: Array,
    ):
        """
        Helper class for efficiently pre-computing multiple penalty
        matrices alongside their ranks and log determinants based on a
        grid of "omega" anisotropy parameters in ``weights``.

        Parameters
        ----------
        pen1, pen2
            Penalty matrices.
        weights
            Anisotropy parameters; weights that must be
            ``0.0 < omega < 1.0``.
        Z
            Reparameterisation matrix. Will be applied to the penalties
            to ensure identifiability. If ``None``, falls back to an
            identity matrix, i.e. no reparameterisation.
        """
        _assert_square(pen1)
        _assert_square(pen2)

        if not np.all(weights < 1.0) and np.all(weights > 0.0):
            raise ValueError(
                "All values in `weights` must be larger than 0 and smaller than 1."
            )

        self.weights = weights
        self.penalties = list()
        self.ranks = list()
        self.log_pdets = list()

        for weight in self.weights:
            penalty = tp_penalty(pen1=pen1, pen2=pen2, weight=weight, augment=True)
            penalty = Z.T @ penalty @ Z
            rk_ldet = _RankDet(penalty)

            self.penalties.append(penalty)
            self.ranks.append(rk_ldet.rank)
            self.log_pdets.append(rk_ldet.log_pdet)


class PenaltyGroupTP(Group):
    def __init__(
        self,
        name: str,
        pen1: Array,
        pen2: Array,
        weights: Array,
        Z: Data | None = None,
    ):
        Zval = (
            Z.value
            if Z is not None
            else np.eye(pen1.shape[-1] * pen2.shape[-1]).astype(np.float32)
        )

        self.nparam = Zval.shape[-1]
        """Number of *identified* parameters associated with this penalty group."""

        grid = _PenaltyGridTP(pen1, pen2, weights=weights, Z=Zval)

        weights = jnp.array(grid.weights)
        self.weight_grid = Var(weights, name=name + "_weight_grid")
        """Pre-defined grid of possible values for the weight parameter."""

        probs = Data(
            np.full(
                len(grid.weights), fill_value=1 / len(grid.weights), dtype=np.float32
            )
        )
        self.probs = Var(probs, name=name + "_weight_probs")
        """Prior probability for the values in :attr:`.self.weight_grid`."""

        weights_prior = Dist(
            tfd.FiniteDiscrete, outcomes=self.weight_grid, probs=self.probs
        )
        self.weight = Param(
            np.quantile(weights, 0.5),
            distribution=weights_prior,
            name=name + "_weight",
        )
        """The anisotropy weight parameter (omega in Kneib's notation)."""

        # Lookup nodes for cached penalty matrix with rank and log determinant

        self.penalty = Var(
            LookUpCalculator(
                mapping=dict(zip(grid.weights, grid.penalties)), key_node=self.weight
            ),
            name=name,
        )
        """The penalty matrix."""

        self.rank = Data(np.unique(grid.ranks), _name=name + "_rank")
        """The rank of the penalty matrix."""

        self.log_pdet = LookUp(
            dict(zip(grid.weights, grid.log_pdets)),
            key_node=self.weight,
            name=name + "_log_pdet",
        )
        """The pseudo-log-determinant of the penalty matrix."""

        super().__init__(
            name=name,
            probs=self.probs,
            weight_grid=self.weight_grid,
            weight=self.weight,
            penalty=self.penalty,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )


# ----------------------------------------------------------------------
# Basis matrices for univariate splines
# ----------------------------------------------------------------------


class BSplineBasis(Var):
    """A design matrix of B-spline basis function evaluations."""

    observed = True

    def __init__(
        self,
        value: Array,
        nparam: int,
        order: int = 3,
        name: str | None = None,
        knots: Array | None = None,
    ) -> None:
        # TODO: Make sure this is correct after update to liesel-internal
        if knots is None:
            knots = kn(value, order=order, n_params=nparam)
        B = SplineDesign(value, knots=knots, order=order)
        assert B.shape[-1] == nparam

        super().__init__(value=B, name=name)

        self.Z = np.eye(nparam, dtype=np.float32)
        """Reparameterisation matrix for identifiability."""

        self.obs_value = value
        """The observed covariate values."""

        self.nparam = nparam
        """Number of *identified* parameters associated with this covariate."""

        self.knots = knots
        """Array of knots that were used to create the BasisMatrix."""

        self.order = order
        """Order of the B-splines used in this basis matrix."""

    def reparam(self, Z: Array) -> BSplineBasis:
        """
        Adds a reparameterisation matrix for identifiability.
        Updates the number of parameters, since usually this number
        decreases with identifiability constraints.

        Example::
            import numpy as np
            from bctm.gam import p_splines as ps
            from bctm.gam.constraints import sumzero

            x = np.random.uniform(size=20)
            B = ps.BSplineBasis(x, nparam=7)
            B.reparam(sumzero(B.value))
        """
        self.Z = Z
        self.value = self.value @ self.Z  # type: ignore
        self.nparam = self.value.shape[-1]
        return self

    def bs(self, value: Array | float | None) -> Array:
        """
        Evaluate the B-spline basis functions with :attr:`.knots` and
        :attr:`.order` for new values.
        """
        if value is None:
            return self.value
        value = np.atleast_1d(np.array(value, dtype=np.float32))
        return SplineDesign(value, knots=self.knots, order=self.order) @ self.Z

    def d(self, value: Array | float | None = None) -> Array:
        """
        Evaluate the derivative of the B-spline basis functions with
        :attr:`.knots` and :attr:`.order` for new values.

        If ``value=None``, evaluate the derivative at :attr:`.obs_value`.
        """
        value = value if value is not None else self.obs_value
        value = np.atleast_1d(np.array(value, dtype=np.float32))
        return SplineDesign_d(value, knots=self.knots, order=self.order) @ self.Z


class BSplineBasisCentered(BSplineBasis):
    """A *centered* design matrix of B-spline basis function evaluations."""

    observed = True

    def __init__(
        self,
        value: Array,
        nparam: int,
        order: int = 3,
        name: str | None = None,
    ) -> None:
        # TODO: Make sure this is correct after update to liesel-internal
        knots = kn(value, order=order, n_params=nparam)
        B = SplineDesign(value, knots=knots, order=order)
        colmeans = B.mean(axis=0)
        Bc = B - colmeans

        super(BSplineBasis, self).__init__(value=Bc, name=name)

        self.colmeans = colmeans
        """Column means of uncentered basis matrix."""

        self.Z = np.eye(nparam, dtype=np.float32)
        """
        Reparameterisation matrix for identifiability; identity matrix
        for centered bases.
        """

        self.obs_value = value
        """The observed covariate values."""

        self.nparam = nparam
        """Number of *identified* parameters associated with this covariate."""

        self.knots = knots
        """Array of knots that were used to create the BasisMatrix."""

        self.order = order
        """Order of the B-splines used in this basis matrix."""

    def bs(self, value: Array | None) -> Array:
        if value is None:
            return self.value
        value = np.atleast_1d(np.array(value, dtype=np.float32))
        X = SplineDesign(value, knots=self.knots, order=self.order)
        return (X - self.colmeans) @ self.Z


_kron_rowwise_helper = jax.vmap(jnp.kron, (0, 0), 0)


def kron_rowwise(a: Array, b: Array):
    try:
        return _kron_rowwise_helper(a, b)
    except ValueError as e:
        if len(b) == 1 and b == 1:
            return a
        raise e


def _assert_shape(a, b):
    a_shape = 1 if isinstance(a, float) else a.shape
    b_shape = 1 if isinstance(b, float) else b.shape

    if not a_shape == b_shape:
        raise ValueError(
            "The two arrays must be of equal shape or scalar. Got"
            f" a.shape={a.shape} and b.shape={b.shape}."
        )


class TPBasis(Var):
    """
    A design matrix of the row-wise tensor product of two marginal
    :class:`.BSplineBasis` objects.
    """

    observed = True

    def __init__(
        self, A: BSplineBasis, B: BSplineBasis, name: str | None = None
    ) -> None:
        C = kron_rowwise(A.value, B.value)
        super().__init__(C, name=name)

        self.A = A
        """First marginal :class:`.BSplineBasis`."""
        self.B = B
        """Second marginal :class:`.BSplineBasis`."""
        self.nparam = C.shape[-1]
        """Number of *identified* parameters associated with this design matrix."""

        self.Z = np.eye(self.nparam, dtype=np.float32)
        """
        Reparameterisation matrix for identifiability. Not needed, if the marginal
        design matrices already have an identifiability constraint. Initialised as
        the identity matrix; a new one can be applied with :meth:`.reparam`.
        """

    def bs(self, x: tuple[Array | float, Array | float] | None) -> Array:
        """Evaluate the basis at fixed values."""
        if x is None:
            return self.value  # type: ignore
        a, b = x

        a = a if a is not None else self.A.obs_value
        b = b if b is not None else self.B.obs_value

        if isinstance(b, float) and not isinstance(a, float):
            b = np.full(a.shape, b, dtype=np.float32)

        _assert_shape(a, b)
        return kron_rowwise(self.A.bs(a), self.B.bs(b)) @ self.Z

    def d(
        self, x: tuple[Array | float | None, Array | float | None] | None = (None, None)
    ) -> Array:
        """Derivative with respect to A."""
        return self.da(x=x)

    def da(
        self, x: tuple[Array | float | None, Array | float | None] | None = (None, None)
    ) -> Array:
        """Derivative with respect to A."""

        x = (None, None) if x is None else x

        a, b = x

        a = a if a is not None else self.A.obs_value
        if isinstance(b, float) and not isinstance(a, float):
            b = np.full(a.shape, b, dtype=np.float32)
        b = b if b is not None else self.B.obs_value

        _assert_shape(a, b)
        return kron_rowwise(self.A.d(a), self.B.bs(b)) @ self.Z

    def db(
        self, x: tuple[Array | float | None, Array | float | None] | None = (None, None)
    ) -> Array:
        """Derivative with respect to B."""
        x = (None, None) if x is None else x

        a, b = x
        a = a if a is not None else self.A.obs_value
        if isinstance(b, float) and not isinstance(a, float):
            b = np.full(a.shape, b, dtype=np.float32)
        b = b if b is not None else self.B.obs_value
        _assert_shape(a, b)
        return kron_rowwise(self.A(a), self.B.d(b)) @ self.Z

    def reparam(self, Z: Array) -> TPBasis:
        """
        Adds a reparameterisation matrix for identifiability.
        Updates the number of parameters, since usually this number
        decreases with identifiability constraints.
        """
        self.Z = Z
        self.value = self.value @ Z  # type: ignore
        self.nparam = self.value.shape[-1]
        return self


# ----------------------------------------------------------------------
# Spline coefficient
# ----------------------------------------------------------------------


class SplineCoef(Var):
    """
    Spline coefficient with a multivariate normal prior.

    The prior is centered around zero and defined by the ``var_param`` and the
    ``penalty.penalty`` penalty matrix. The penalty matrix may be rank-deficient.
    """

    parameter = True

    def __init__(
        self,
        var_param: Param,
        penalty: PenaltyGroup | PenaltyGroupTP,
        name: str = "",
    ):
        # Regression coefficient node
        self.m = Var(0.0, name=name + "_prior_mean")
        """Var for the prior mean."""

        prior = Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=self.m,
            pen=penalty.penalty,
            var=var_param,
            log_pdet=penalty.log_pdet,
            rank=penalty.rank,
        )
        start_value = np.full(penalty.nparam, self.m.value, dtype=np.float32)

        super().__init__(start_value, prior, name=name)
        self.value_node.monitor = True


# ----------------------------------------------------------------------
# Complete spline groups
# ----------------------------------------------------------------------


class PSpline(Group):
    """A complete univariate P-Spline."""

    return_kernels = True

    def __init__(
        self,
        name: str,
        x: Array,
        nparam: int,
        a: float,
        b: float,
        order: int = 3,
        Z: Array | None = None,
    ) -> None:
        B = BSplineBasis(x, nparam=nparam, order=order, name=name + "_B")
        self.Z = Data(Z) if Z is not None else Data(constraints.sumzero(B.value))
        """Reparameterisation matrix for identifiability."""

        self.X = B.reparam(self.Z.value)
        """The matrix of basis function evaluations."""

        self.nparam = self.X.value.shape[-1]
        """Number of *identified* parameters associated with this Pspline."""

        self.var_group = IGVariance(name=name + "_igvar", a=a, b=b, start_value=10000.0)
        """The variance parameter group."""

        self.pen_group = PenaltyGroup(name + "_penalty", pen=pen(nparam), Z=self.Z)
        """The penalty group."""

        self.coef = SplineCoef(
            self.var_group.var_param, self.pen_group, name=name + "_coef"
        )
        """The coefficient."""

        self.smooth = Var(Calc(jnp.dot, self.X, self.coef), name=name)
        """The smooth variable."""

        self.sampled_params = [self.coef.name]
        """Names of the params that can be sampled with samplers like NUTS or IWLS."""

        contents = self.var_group.nodes_and_vars | self.pen_group.nodes_and_vars

        super().__init__(
            name, Z=self.Z, X=self.X, smooth=self.smooth, coef=self.coef, **contents
        )

    def _gibbs_kernels(self) -> list[GibbsKernel]:
        var_kernel = gibbs.igvar_gibbs_kernel(self)
        return [var_kernel]

    def ppeval(self, samples: dict, x: Array | float | None = None) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        return ppeval(self, samples, x)


class PSplineTP(Group):
    """
    A complete tensor-product interaction P-Spline.

    Parameters
    ----------
    Z
        Reparameterisation matrix. If a tuple is supplied, the reparameterisation will
        be applied to the marginal design matrices, resembling the "ti" behavior of
        mgcv. If an array is supplied, the the reparameterisation will be applied the
        the tensor-product design matrix, resembling the "tp" behavior of mgcv. If a
        value is `None`, this group automatically uses a sum-to-zero constraint for the
        corresponding marginal or tensor-product design matrix.
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
        Z: tuple[Array | None, Array | None] | Array = (None, None),
        knots: tuple[Array | None, Array | None] = (None, None),
    ) -> None:

        A = BSplineBasis(x[0], nparam[0], order=order, name=name + "_A", knots=knots[0])
        B = BSplineBasis(x[1], nparam[1], order=order, name=name + "_B", knots=knots[1])

        nparam1 = A.value.shape[-1]
        nparam2 = B.value.shape[-1]

        if isinstance(Z, tuple):
            Z1 = Z[0] if Z[0] is not None else constraints.sumzero(A.value)
            Z2 = Z[1] if Z[1] is not None else constraints.sumzero(B.value)

            A.reparam(Z1)
            B.reparam(Z2)

            Z = np.kron(Z1, Z2)
            C = TPBasis(A, B, name=name + "_C")
        else:
            C = TPBasis(A, B, name=name + "_C")
            Z = Z if Z is not None else constraints.sumzero(C.value)
            C.reparam(Z)

        self.Z = Data(Z)
        """Reparameterisation matrix for identifiability."""

        self.X = C
        """Basis matrix / design matrix of this smooth."""

        self.nparam = self.X.value.shape[-1]
        """Number of *identified* parameters associated with this Pspline."""

        self.var_group = IGVariance(name=name + "_igvar", a=a, b=b, start_value=10000.0)
        """The variance parameter group."""

        weights = (
            weights
            if weights is not None
            else np.linspace(0.001, 0.999, 15, dtype=np.float32)
        )
        self.pen_group = PenaltyGroupTP(
            name + "_penalty",
            pen1=pen(nparam1),
            pen2=pen(nparam2),
            Z=self.Z,
            weights=weights,
        )
        """The penalty group."""

        self.coef = SplineCoef(
            self.var_group.var_param, self.pen_group, name=name + "_coef"
        )
        """The coefficient."""

        self.smooth = Var(Calc(jnp.dot, self.X, self.coef), name=name)
        """The smooth variable."""

        self.sampled_params = [self.coef.name]
        """Names of the params that can be sampled with samplers like NUTS or IWLS."""

        contents = self.var_group.nodes_and_vars | self.pen_group.nodes_and_vars
        super().__init__(
            name, Z=self.Z, X=self.X, coef=self.coef, smooth=self.smooth, **contents
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
        return ppeval(self, samples, x)


def ppeval(
    group,
    samples,
    x: Array | float | None | tuple[Array | float | None, Array | None],
) -> Array:
    """
    Return posterior predictive evaluation of this smooth given an array of samples.

    Uses ``group.X.value`` if ``x=None``.
    """
    coef_samples = samples[group.coef.name]
    X = group.X.bs(x)
    smooth = np.tensordot(X, coef_samples, axes=([1], [-1]))
    return np.moveaxis(smooth, 0, -1)
