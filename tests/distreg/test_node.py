import numpy as np
import pytest

from liesel_bctm.distreg import node as nd

x = np.random.uniform(-2, 2, size=(20, 2))


def test_svar() -> None:
    y = np.linspace(0, 10, 100)
    resp = nd.SVar(y)
    assert resp.mean == pytest.approx(5.0)
    assert resp.scale.round() == pytest.approx(3.0)
    assert resp.value.mean() == pytest.approx(0.0, abs=1e-03)
    assert resp.value.std().round() == pytest.approx(1.0)


def test_lin() -> None:
    lin = nd.Lin("test", x, m=0.0, s=100.0)

    assert lin is not None
    assert np.allclose(lin.coef.value, np.zeros(1))


def test_lin_ppeval() -> None:
    lin = nd.Lin("test", x, m=0.0, s=100.0)

    coef_samples = np.random.normal(
        size=(3, 10, 2)
    )  # shape: (nchains, nsamples, nparam)
    # 1 param less because of identifiability constraint
    samples = {lin.coef.name: coef_samples}

    prediction = lin.ppeval(samples, x=np.array([[1.0, 2.0]]))
    assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

    prediction = lin.ppeval(samples, x=np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]]).T)
    assert prediction.shape == (3, 10, 3)  # (nchains, nsamples, nvalues)
