"""
Functionality for the penalty in a transforming tensor product.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.nn import softplus
from liesel.goose import GibbsKernel
from liesel.model import Calc, Data, Dist, Var

from .custom_types import Array
from .distreg import constraints, gibbs
from .distreg import psplines as ps
from .distreg.mi_splines import _apply_positive_tranformation, _cumsum, mi_pen
from .distreg.node import Group
from .distreg.psplines import _RankDet, pen
from .liesel_internal import splines
from .liesel_internal.lookup import LookUp, LookUpCalculator

SplineDesign = splines.build_design_matrix_b_spline
SplineDesign_d = splines.build_design_matrix_b_spline_derivative
kn = splines.create_equidistant_knots


def cov_pen(nparam1: int, nparam2: int) -> Array:
    pen2_1d = pen(nparam2, diff=1)
    pen2_2d = pen(nparam2, diff=2)
    second_difference_pen = np.kron(np.array([1] + [0] * (nparam1 - 1)), pen2_2d)

    Imat = np.hstack((np.zeros((nparam1 - 1, 1)), np.eye(nparam1 - 1)))
    first_difference_pen = np.kron(Imat, pen2_1d)
    penalty2 = np.vstack((second_difference_pen, first_difference_pen))
    return penalty2


def cov_pen_exp2(nparam1: int, nparam2: int) -> Array:
    pen2_1d = pen(nparam2, diff=1)
    pen2_2d = pen(nparam2, diff=2)
    second_difference_pen = np.kron(np.array([1] + [0] * (nparam1 - 1)), pen2_2d)

    Imat = np.hstack((np.zeros((nparam1 - 1, 1)), np.eye(nparam1 - 1)))
    first_difference_pen = np.kron(Imat, pen2_1d)
    penalty2 = np.vstack((second_difference_pen, first_difference_pen))
    return penalty2


def tp_penalty(
    nparam1: int, nparam2: int, weight: float, augment: bool = False
) -> Array:
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
    pen1 = mi_pen(nparam1)
    penalty1 = np.kron(pen1, np.eye(nparam2))
    penalty2 = cov_pen(nparam1, nparam2)

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


def tp_penalty_exp2(
    nparam1: int, nparam2: int, weight: float, augment: bool = False
) -> Array:
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
    pen1 = mi_pen(nparam1)

    penalty1 = np.kron(pen1, np.eye(nparam2))
    penalty1 = penalty1[nparam2:, nparam2:]

    pen2_1d = pen(nparam2, diff=1)
    penalty2 = np.kron(np.eye(nparam1 - 1), pen2_1d)

    penalty = np.array(weight * penalty1 + (1.0 - weight) * penalty2, dtype=np.float32)

    if augment:
        # Add small values to diag to increase numerical stability
        # This is not EXACTLY as Carlan (2022) described in the paper, because this
        # penalty
        # matrix here gets multiplied with 1/tau2 later.
        # Carlan (2022) seem to augment the full precision matrix, not only the
        # penalty matrix.
        di = np.diag_indices(penalty.shape[-1])
        np.fill_diagonal(penalty, penalty[di] + 1e-5)
        # penalty = penalty.at[di].set(penalty[di] + 1e-5)

    return penalty


class _PenaltyGridTP:
    def __init__(
        self,
        nparam1: int,
        nparam2: int,
        weights: Array,
        Z: Array,
    ):
        """
        Helper class for efficiently pre-computing multiple penalty
        matrices alongside their ranks and log determinants based on a
        grid of "omega" anisotropy parameters in ``weights``.

        Parameters
        ----------
        weights
            Anisotropy parameters; weights that must be
            ``0.0 < omega < 1.0``.
        Z
            Reparameterisation matrix. Will be applied to the penalties
            to ensure identifiability. If ``None``, falls back to an
            identity matrix, i.e. no reparameterisation.
        """
        if not np.all(weights < 1.0) and np.all(weights > 0.0):
            raise ValueError(
                "All values in `weights` must be larger than 0 and smaller than 1."
            )

        self.weights = weights
        self.penalties = list()
        self.ranks = list()
        self.log_pdets = list()

        for weight in self.weights:
            penalty = tp_penalty(nparam1, nparam2, weight=weight, augment=True)
            penalty = Z.T @ penalty @ Z
            rk_ldet = _RankDet(penalty)

            self.penalties.append(penalty)
            self.ranks.append(rk_ldet.rank)
            self.log_pdets.append(rk_ldet.log_pdet)


class _PenaltyGridTPExp2:
    def __init__(
        self,
        nparam1: int,
        nparam2: int,
        weights: Array,
        Z: Array,
    ):
        """
        Helper class for efficiently pre-computing multiple penalty
        matrices alongside their ranks and log determinants based on a
        grid of "omega" anisotropy parameters in ``weights``.

        Parameters
        ----------
        weights
            Anisotropy parameters; weights that must be
            ``0.0 < omega < 1.0``.
        Z
            Reparameterisation matrix. Will be applied to the penalties
            to ensure identifiability. If ``None``, falls back to an
            identity matrix, i.e. no reparameterisation.
        """
        if not np.all(weights < 1.0) and np.all(weights > 0.0):
            raise ValueError(
                "All values in `weights` must be larger than 0 and smaller than 1."
            )

        self.weights = weights
        self.penalties = list()
        self.ranks = list()
        self.log_pdets = list()

        for weight in self.weights:
            penalty = tp_penalty_exp2(nparam1, nparam2, weight=weight, augment=True)
            penalty = Z.T @ penalty @ Z
            rk_ldet = _RankDet(penalty)

            self.penalties.append(penalty)
            self.ranks.append(rk_ldet.rank)
            self.log_pdets.append(rk_ldet.log_pdet)


class PenaltyGroupTP(Group):
    def __init__(
        self,
        name: str,
        nparam1: int,
        nparam2: int,
        weights: Array,
        Z: Data | None = None,
    ):
        Zval = (
            Z.value if Z is not None else np.eye(nparam1 * nparam2).astype(np.float32)
        )

        self.nparam = Zval.shape[-1]
        """Number of *identified* parameters associated with this penalty group."""

        grid = _PenaltyGridTP(nparam1, nparam2, weights=weights, Z=Zval)

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
        self.weight = Var.new_param(
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


class PenaltyGroupTPExp2(Group):
    def __init__(
        self,
        name: str,
        nparam1: int,
        nparam2: int,
        weights: Array,
        Z: Data | None = None,
    ):
        Zval = (
            Z.value if Z is not None else np.eye(nparam1 * nparam2).astype(np.float32)
        )

        self.nparam = Zval.shape[-1]
        """Number of *identified* parameters associated with this penalty group."""

        grid = _PenaltyGridTPExp2(nparam1, nparam2, weights=weights, Z=Zval)

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
        self.weight = Var.new_param(
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
    ) -> None:
        A = ps.BSplineBasis(x[0], nparam[0], order=order, name=name + "_A")
        B = ps.BSplineBasis(x[1], nparam[1], order=order, name=name + "_B")

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
        self.pen_group = PenaltyGroupTPExp2(
            name + "_penalty",
            nparam1=nparam1,
            nparam2=nparam2,
            Z=self.Z,
            weights=weights,
        )
        """The penalty group."""

        self.coef = ps.SplineCoef(
            self.var_group.var_param,
            self.pen_group,  # type: ignore
            name=name + "_coef",
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
