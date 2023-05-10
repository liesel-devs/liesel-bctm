from typing import Iterator

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.model import Calc, GraphBuilder, Obs, Var

from liesel_bctm.dist import TDist
from liesel_bctm.distreg import mi_splines as mi

n = 20
y = np.random.uniform(size=n)
Ndist = tfd.Normal(loc=0.0, scale=1.0)


@pytest.fixture
def mips() -> Iterator[mi.MIPSpline]:
    yield mi.MIPSpline("mips", x=y, nparam=7, a=2.0, b=0.5)


@pytest.fixture
def ct(mips: mi.MIPSpline) -> Iterator[Var]:
    yield mips.smooth


@pytest.fixture
def ctd(mips: mi.MIPSpline) -> Iterator[Var]:
    basis_derivative = Obs(mips.X.d(), name="Bd")
    positive_coef = mips.positive_coef
    yield Var(Calc(jnp.dot, basis_derivative, positive_coef), name="ctd")


class TestTransformationDist:
    def test_init(self, ct: Var, ctd: Var) -> None:
        GraphBuilder().add(ct, ctd).build_model()
        dist = TDist(ct=ct, ctd=ctd, refdist=Ndist)
        dist.update()
        assert dist.value.shape[-1] == n
        assert dist.log_prob.shape[-1] == n
        assert np.allclose(dist.value, dist.log_prob)
        assert not np.any(np.isnan(dist.log_prob))

    def test_as_dist(self, ct: Var, ctd: Var) -> None:
        dist = TDist(ct=ct, ctd=ctd, refdist=Ndist)
        yvar = Obs(y, distribution=dist, name="y")

        m = GraphBuilder().add(yvar).build_model()  # updates everything

        assert np.allclose(yvar.log_prob, dist.log_prob)
        assert (m.log_prior + m.log_lik) == pytest.approx(m.log_prob)
