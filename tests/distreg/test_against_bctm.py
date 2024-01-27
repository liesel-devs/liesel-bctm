from collections import defaultdict

import numpy as np
import pandas as pd
from liesel_internal import splines

from liesel_bctm.distreg import constraints
from liesel_bctm.distreg import mi_splines as mi
from liesel_bctm.distreg import psplines as ps

SplineDesign = splines.build_design_matrix_b_spline
SplineDesign_d = splines.build_design_matrix_b_spline_derivative
kn = splines.create_equidistant_knots

df = pd.read_csv("tests/files/framingham.csv", index_col=None)
bctm_Z = pd.read_csv(
    "tests/files/bctm_reparam.csv",
    index_col=None,
    header=None,
    dtype=defaultdict(np.float32),
).to_numpy()
bctm_basis = pd.read_csv(
    "tests/files/bctm_design_matrix.csv",
    index_col=None,
    header=None,
    dtype=defaultdict(np.float32),
).to_numpy()
bctm_Ky = pd.read_csv(
    "tests/files/bctm_penalty_y.csv",
    index_col=None,
    header=None,
    dtype=defaultdict(np.float32),
).to_numpy()
bctm_Kx = pd.read_csv(
    "tests/files/bctm_penalty_x.csv",
    index_col=None,
    header=None,
    dtype=defaultdict(np.float32),
).to_numpy()
bctm_knots1 = pd.read_csv(
    "tests/files/bctm_knots1.csv",
    index_col=None,
    header=None,
    dtype=defaultdict(np.float32),
).to_numpy()
bctm_knots2 = pd.read_csv(
    "tests/files/bctm_knots2.csv",
    index_col=None,
    header=None,
    dtype=defaultdict(np.float32),
).to_numpy()
q1 = 4
q2 = 4


def test_reparameterization_matrix() -> None:
    myZ = constraints.mi_sumzero(q1, q2)
    assert np.allclose(myZ, bctm_Z)


def test_design_matrix() -> None:
    A = ps.BSplineBasis(df.cholst.to_numpy(), nparam=4)
    B = ps.BSplineBasis(df.age.to_numpy(), nparam=4)
    Z = constraints.mi_sumzero(q1, q2)
    S = mi._cumsum(q1, q2)
    C = ps.TPBasis(A, B)
    C.reparam(S @ Z)

    assert not np.allclose(C.value, bctm_basis)  # the bases are not exactly equal
    assert np.all(np.abs(C.value - bctm_basis) < 0.1)  # but very close

    # the unequality stems from slightly differently defined knots
    # if I use Manus bctm knots, we have equality
    A.value = SplineDesign(df.cholst.to_numpy(), knots=np.squeeze(bctm_knots1), order=3)
    B.value = SplineDesign(df.age.to_numpy(), knots=np.squeeze(bctm_knots2), order=3)

    C = ps.TPBasis(A, B)
    C.reparam(S @ Z)

    assert np.allclose(C.value, bctm_basis)


def test_penalty_y() -> None:
    pen1 = mi.mi_pen(q1)
    penalty1 = np.kron(pen1, np.eye(q2))
    assert np.allclose(penalty1, bctm_Ky)


def test_penalty_x() -> None:
    """
    Manus uses a different penalty for the covariate. He uses 2nd order only for
    the untransformed coefficients. For the transformed ones, he uses 1st order.
    """
    pen2 = ps.pen(q2)
    penalty2 = np.kron(np.eye(q1), pen2)
    assert not np.allclose(penalty2, bctm_Kx)
