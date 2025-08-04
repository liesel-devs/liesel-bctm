from collections.abc import Iterator

import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_bctm.custom_types import Array
from liesel_bctm.distreg import constraints
from liesel_bctm.distreg import psplines as ps

x = np.random.uniform(size=20)


def test_ig_variance() -> None:
    igvar = ps.IGVariance("test", a=2.0, b=0.5)
    ig = tfd.InverseGamma(2.0, 0.5)

    assert isinstance(igvar.var_param, lsl.Var)
    assert igvar.var_param.value == 100.0
    assert igvar.var_param.update().log_prob == ig.log_prob(100.0)


def test_penalty_group() -> None:
    K = ps.pen(5)
    pen = ps.PenaltyGroup("test", K)

    assert pen.penalty.value.shape == (5, 5)


class TestPenaltyGroupTP:
    def test_init(self) -> None:
        K = ps.pen(5)
        weights = np.linspace(0.001, 0.999, 3)
        pen = ps.PenaltyGroupTP("test", K, K, weights=weights)

        assert pen.weight.update().log_prob
        assert pen.penalty.update().value.shape == (25, 25)


@pytest.fixture
def B() -> Iterator[ps.BSplineBasis]:
    yield ps.BSplineBasis(x, nparam=7)


class TestBasisMatrix:
    def test_init(self, B: ps.BSplineBasis) -> None:
        assert np.all(B.obs_value == x)

    def test_nparam(self) -> None:
        nparam = 7
        B = ps.BSplineBasis(x, nparam=nparam)
        assert B.value.shape[-1] == nparam

    def test_rowsum(self, B: ps.BSplineBasis) -> None:
        assert np.allclose(B.value.sum(axis=1), 1.0)

    def test_bs(self, B: ps.BSplineBasis) -> None:
        assert B.bs(0.5).shape == (1, 7)
        assert np.allclose(B.bs(x[10]), B.value[10, :])
        assert np.allclose(B.bs(x), B.value)

    def test_d(self, B: ps.BSplineBasis) -> None:
        assert B.d(x).shape == B.value.shape

    def test_basis_matrix_centered(self) -> None:
        B = ps.BSplineBasisCentered(x, nparam=7)
        assert np.allclose(B.value.mean(axis=0), 0.0, atol=1e-07)

    def test_tp_basis(self, B: ps.BSplineBasis) -> None:
        C = ps.TPBasis(B, B)
        assert C.nparam == B.nparam**2
        assert np.allclose(C.value, ps.kron_rowwise(B.value, B.value))


class TestSplineCoef:
    def test_init(self) -> None:
        igvar = ps.IGVariance("test", a=2.0, b=0.5)
        pen = ps.PenaltyGroup("test", ps.pen(7))
        coef = ps.SplineCoef(igvar.var_param, pen)
        assert coef.value is not None

    def test_dimension(self) -> None:
        igvar = ps.IGVariance("test", a=2.0, b=0.5)
        pen = ps.PenaltyGroup("test", ps.pen(7))
        coef = ps.SplineCoef(igvar.var_param, pen)
        assert coef.value.shape == (7,)

    def test_with_tp_penalty(self) -> None:
        igvar = ps.IGVariance("test", a=2.0, b=0.5)

        K = ps.pen(5)
        weights = np.linspace(0.001, 0.999, 3)
        pen = ps.PenaltyGroupTP("test", K, K, weights=weights)

        for var in pen.vars.values():
            var.update()  # all vars in the group need to be updated first

        coef = ps.SplineCoef(igvar.var_param, pen)
        assert coef.value.shape == (25,)

    def test_sumzero_constraint(self, B: ps.BSplineBasis) -> None:
        igvar = ps.IGVariance("test", a=2.0, b=0.5)

        B = ps.BSplineBasis(x, nparam=7)
        Z = constraints.sumzero(B.value)
        B = B.reparam(Z)

        pen = ps.PenaltyGroup("test", ps.pen(7), Z=lsl.Data(Z))

        coef = ps.SplineCoef(igvar.var_param, pen)
        coef.value = np.random.uniform(-2, 2, size=B.nparam)

        assert coef.value.shape == (B.nparam,)
        assert coef.value.shape == (6,)
        assert np.sum(B.value @ coef.value) == pytest.approx(0.0, abs=1e-05)


class TestPSpline:
    def test_pspline(self):
        pspline = ps.PSpline("test", x=x, nparam=7, a=2.0, b=0.5)
        assert pspline is not None

    def test_pspline_model(self):
        pspline = ps.PSpline("test", x=x, nparam=7, a=2.0, b=0.5)
        gb = lsl.GraphBuilder().add(pspline.smooth)
        model = gb.build_model()
        assert model is not None

    def test_pspline_ti(self):
        tp = ps.PSplineTP("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5, Z=(None, None))
        assert tp is not None
        assert tp.nparam == 36

    def test_pspline_tp(self):
        tp = ps.PSplineTP("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5, Z=None)
        assert tp is not None
        assert tp.nparam == 48

    def test_pspline_tp_model(self):
        tp = ps.PSplineTP("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5)
        gb = lsl.GraphBuilder().add(tp.smooth)
        model = gb.build_model()
        assert model is not None

    def test_ppeval(self):
        pspline = ps.PSpline("test", x=x, nparam=7, a=2.0, b=0.5)
        coef_samples = np.random.normal(
            size=(3, 10, 6)
        )  # shape: (nchains, nsamples, nparam)
        # 1 param less because of identifiability constraint
        samples = {pspline.coef.name: coef_samples}

        prediction = pspline.ppeval(samples, x=0.5)
        assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

        prediction = pspline.ppeval(samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

        aggregated_samples = {pspline.coef.name: coef_samples.mean(axis=1)}
        shape = aggregated_samples[pspline.coef.name].shape
        # (nchains, nparam)
        assert shape == (3, 6)

        prediction = pspline.ppeval(aggregated_samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (
            3,
            2,
        )  # (nchains, nsamples (=1; squeezed out), nvalues)

        aggregated_samples = {pspline.coef.name: coef_samples.mean(axis=(0, 1))}
        shape = aggregated_samples[pspline.coef.name].shape
        # (nparam,)
        assert shape == (6,)

        prediction = pspline.ppeval(aggregated_samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (2,)  # (nvalues, )


@pytest.fixture
def tp() -> Iterator[ps.PSplineTP]:
    yield ps.PSplineTP("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5)


@pytest.fixture
def samples(tp: ps.PSplineTP) -> Iterator[dict[str, Array]]:
    coef_samples = np.random.normal(
        size=(3, 10, 36)
    )  # shape: (nchains, nsamples, nparam)
    # 1 param less because of identifiability constraint
    samples = {tp.coef.name: coef_samples}
    yield samples


class TestPSPlineTPPosteriorPred:
    def test_ppeval_both_scalars(self, tp: ps.PSplineTP, samples: dict[str, Array]):
        prediction = tp.ppeval(samples, x=(0.5, 0.4))
        assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

    def test_ppeval_both_arrays(self, tp: ps.PSplineTP, samples: dict[str, Array]):
        a = np.array([0.3, 0.6])
        b = np.array([0.2, 0.8])
        prediction = tp.ppeval(samples, x=(a, b))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

    def test_ppeval_one_scalar(self, tp: ps.PSplineTP, samples: dict[str, Array]):
        a = np.array([0.3, 0.6])
        b = 0.2
        prediction = tp.ppeval(samples, x=(a, b))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

    def test_ppeval_both_arrays_different_length(
        self, tp: ps.PSplineTP, samples: dict[str, Array]
    ):
        with pytest.raises(ValueError, match="equal shape"):
            a = np.array([0.3, 0.6])
            b = np.array([0.2, 0.8, 0.7])
            tp.ppeval(samples, x=(a, b))
