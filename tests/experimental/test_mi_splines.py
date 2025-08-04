from collections.abc import Iterator

import jax
import liesel.model as lsl
import numpy as np
import pytest
import scipy

from liesel_bctm import CTMBuilder, ctm_mcmc
from liesel_bctm.builder import MITEDerivative
from liesel_bctm.custom_types import Array
from liesel_bctm.experimental import mi_splines as exmi

rng = np.random.default_rng(seed=1)

x = rng.uniform(size=20)
x.sort()


@pytest.fixture
def data() -> Iterator[dict[str, Array]]:
    rng = np.random.default_rng(seed=1)

    n = 100
    x = rng.uniform(-2, 2, size=(n, 2))
    X = np.column_stack((np.ones(n), np.sort(x[:, 0]), x[:, 1]))
    coef = np.array([1.0, 0.0, 2.0])
    y = rng.normal(loc=X @ coef, scale=2)
    data = {"y": y, "x1": np.sort(x[:, 0]), "x2": x[:, 1]}
    yield data


@pytest.fixture
def tp() -> Iterator[exmi.MIPSplineTE1NoCovariate2MainEffect]:
    tp = exmi.MIPSplineTE1NoCovariate2MainEffect(
        "test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5
    )
    yield tp


class TestMIPSplineTE1NoCovariate2MainEffect:
    def test_init(self) -> None:
        tp = exmi.MIPSplineTE1NoCovariate2MainEffect(
            "test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5
        )
        assert tp is not None

    def test_increasing(self, tp) -> None:
        lsl.GraphBuilder().add(tp.smooth).build_model()
        diff = np.diff(tp.smooth.value)
        assert np.all(diff >= 0.0)

    def test_positive(self, tp) -> None:
        assert np.allclose(tp.positive_coef.update().value, jax.nn.softplus(0.0))

    def test_coef_shape(self, tp) -> None:
        assert tp.coef.value.shape == (42,)

    def test_penalty_shape(self, tp) -> None:
        assert tp.pen_group.penalty.update().value.shape == (42, 42)


class TestAddRestrictedInteractionToBuilder:
    def test_mite1_restricted(self, data) -> None:
        ctmb = (
            CTMBuilder(data).add_intercept().add_pspline("x2", nparam=7, a=2.0, b=0.5)
        )

        tp = exmi.MIPSplineTE1NoCovariate2MainEffect(
            name="yx2",
            x=(data["y"], data["x2"]),
            nparam=(7, 7),
            a=2.0,
            b=0.5,
        )

        ctmb.pt.append(tp)
        ctmb.ptd.append(MITEDerivative(tp, name="yx2_d"))  # type: ignore
        ctmb.add_groups(tp)
        ctmb = ctmb.add_response("y")

        model = ctmb.build_model()

        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_sample_mite1_restricted(self, data) -> None:
        ctmb = (
            CTMBuilder(data).add_intercept().add_pspline("x2", nparam=7, a=2.0, b=0.5)
        )

        tp = exmi.MIPSplineTE1NoCovariate2MainEffect(
            name="yx2",
            x=(data["y"], data["x2"]),
            nparam=(7, 7),
            a=2.0,
            b=0.5,
        )

        ctmb.pt.append(tp)
        ctmb.ptd.append(MITEDerivative(tp, name="yx2_d"))  # type: ignore
        ctmb.add_groups(tp)
        ctmb = ctmb.add_response("y")

        model = ctmb.build_model()
        eb = ctm_mcmc(model, seed=1, num_chains=2)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=(0, 1)))
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p > 0.05
