from collections.abc import Iterator

import jax
import liesel.model as lsl
import numpy as np
import pytest

from liesel_bctm.custom_types import Array
from liesel_bctm.distreg import mi_splines as mi

rng = np.random.default_rng(seed=1)

x = rng.uniform(size=20)
x.sort()


class TestMIPSpline:
    def test_init(self) -> None:
        mips = mi.MIPSpline("test", x=x, nparam=7, a=2.0, b=0.5)
        assert mips is not None

    def test_increasing(self) -> None:
        mips = mi.MIPSpline("test", x=x, nparam=7, a=2.0, b=0.5)
        lsl.GraphBuilder().add(mips.smooth).build_model()
        diff = np.diff(mips.smooth.value)
        assert np.all(diff >= 0.0)

    def test_increasing_softplus(self) -> None:
        mips = mi.MIPSpline(
            "test", x=x, nparam=7, a=2.0, b=0.5, positive_tranformation=jax.nn.softplus
        )
        lsl.GraphBuilder().add(mips.smooth).build_model()
        diff = np.diff(mips.smooth.value)
        assert np.all(diff >= 0.0)

    def test_init_te1(self) -> None:
        tp = mi.MIPSplineTE1("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5, Z=None)
        assert tp is not None
        assert tp.nparam == 48

    def test_te1_model(self) -> None:
        tp = mi.MIPSplineTE1("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5, Z=None)
        gb = lsl.GraphBuilder().add(tp.smooth)
        model = gb.build_model()
        assert model is not None

    def test_increasing_te1(self) -> None:
        tp = mi.MIPSplineTE1("test", x=(x, x), nparam=(5, 5), a=2.0, b=0.5, Z=None)
        lsl.GraphBuilder().add(tp.smooth).build_model()
        coef = np.insert(tp.positive_coef.value, 0, 0)

        coef_matrix = np.reshape(coef, (5, 5))
        assert np.all(coef_matrix[1:, :] >= 0)

        icoef_matrix = np.reshape(mi._cumsum(5, 5) @ coef, (5, 5))
        assert np.all(np.diff(icoef_matrix, axis=0) >= 0)


class TestMIPSplinePosteriorPred:
    def test_ppeval(self):
        mipspline = mi.MIPSpline("test", x=x, nparam=7, a=2.0, b=0.5)
        # shape: (nchains, nsamples, nparam)
        coef_samples = np.exp(np.random.normal(size=(3, 10, 6)))
        # 1 param less because of identifiability constraint
        samples = {mipspline.coef.name: coef_samples}

        prediction = mipspline.ppeval(samples, x=0.5)
        assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

        prediction = mipspline.ppeval(samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)
        diff = prediction[:, :, 1] - prediction[:, :, 0]
        assert np.all(diff >= 0.0)

        aggregated_samples = {mipspline.coef.name: coef_samples.mean(axis=1)}
        # (nchains, nparam)
        assert aggregated_samples[mipspline.coef.name].shape == (
            3,
            6,
        )

        prediction = mipspline.ppeval(aggregated_samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (3, 2)  # (nvalues, nchains, nsamples)


@pytest.fixture
def tp() -> Iterator[mi.MIPSplineTE1]:
    yield mi.MIPSplineTE1("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5)


@pytest.fixture
def samples(tp: mi.MIPSplineTE1) -> Iterator[dict[str, Array]]:
    coef_samples = np.random.normal(
        size=(3, 10, 48)
    )  # shape: (nchains, nsamples, nparam)
    # 1 param less because of identifiability constraint
    samples = {tp.coef.name: np.exp(coef_samples)}
    yield samples


class TestMIPSplineTE1PosteriorPred:
    def test_ppeval_both_scalars(self, tp: mi.MIPSplineTE1, samples: dict[str, Array]):
        prediction = tp.ppeval(samples, x=(0.5, 0.4))
        assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

    def test_ppeval_both_arrays(self, tp: mi.MIPSplineTE1, samples: dict[str, Array]):
        a = np.array([0.3, 0.6])
        b = np.array([0.2, 0.8])
        prediction = tp.ppeval(samples, x=(a, b))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

    def test_ppeval_one_scalar(self, tp: mi.MIPSplineTE1, samples: dict[str, Array]):
        a = np.array([0.3, 0.6])
        b = 0.2
        prediction = tp.ppeval(samples, x=(a, b))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

    def test_ppeval_both_arrays_different_length(
        self, tp: mi.MIPSplineTE1, samples: dict[str, Array]
    ):
        with pytest.raises(ValueError, match="equal shape"):
            a = np.array([0.3, 0.6])
            b = np.array([0.2, 0.8, 0.7])
            tp.ppeval(samples, x=(a, b))
