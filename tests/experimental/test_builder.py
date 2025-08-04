import numpy as np
import pytest

from liesel_bctm.experimental.builder import ExperimentalCTMBuilder

seeds = np.array([1, 2, 3, 4])
np.random.shuffle(seeds)

rng = np.random.default_rng(seed=seeds[0])

n = 100
x = rng.uniform(-2, 2, size=(n, 2))
X = np.column_stack((np.ones(n), np.sort(x[:, 0]), x[:, 1]))
coef = np.array([1.0, 0.0, 2.0])
y = rng.normal(loc=X @ coef, scale=2)
data = {"y": y, "x1": np.sort(x[:, 0]), "x2": x[:, 1]}


class TestBuilder:
    def test_teprod_exp2(self) -> None:
        ctmb = (
            ExperimentalCTMBuilder(data)
            .add_intercept()
            .add_pspline("x2", nparam=7, a=2.0, b=0.5, name="x")
            .add_teprod_exp2("y", "x2", nparam=(7, 7), a=2.0, b=0.5, name="yx")  # type: ignore
            .add_response("y")
        )

        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)
