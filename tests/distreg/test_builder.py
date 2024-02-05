from typing import Iterator

import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_bctm.distreg.builder import DistRegBuilder, dist_reg_mcmc

seeds = np.array([1, 2, 3, 4])
np.random.shuffle(seeds)

rng = np.random.default_rng(seed=seeds[0])

n = 100
x = rng.uniform(-2, 2, size=(n, 2))
X = np.column_stack((np.ones(n), np.sort(x[:, 0]), x[:, 1]))
coef = np.array([1.0, 0.0, 2.0])
y = rng.normal(loc=X @ coef, scale=2)
data = {"y": y, "x1": np.sort(x[:, 0]), "x2": x[:, 1]}


@pytest.fixture
def drb() -> Iterator[DistRegBuilder]:
    yield DistRegBuilder(data)


@pytest.mark.xfail
class TestDistRegBuilder:
    def test_init(self, drb: DistRegBuilder) -> None:
        assert drb is not None

    def test_lin(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_response("y", tfd.Normal)
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_linear("x1", "x2", m=0.0, s=100.0, param="loc")
        )

        model = drb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_ps(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_pspline("x1", nparam=7, a=2.0, b=0.5, param="loc")
            .add_pspline("x2", nparam=7, a=2.0, b=0.5, param="loc")
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_te(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_teprod_full("x1", "x2", nparam=(7, 7), a=2.0, b=0.5, param="loc")
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_ti(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_intercept(param="loc")
            .add_teprod_interaction(
                "x1", "x2", nparam=(7, 7), a=2.0, b=0.5, param="loc"
            )
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_mips(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_intercept(param="loc")
            .add_pspline_mi("x1", nparam=7, a=2.0, b=0.5, param="loc")
            .add_pspline("x2", nparam=7, a=2.0, b=0.5, param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_mite1(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_teprod_mi1_full(
                "x1", "x2", nparam=(7, 7), a=2.0, b=0.5, param="loc", name="mite"
            )
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)
        assert model.groups()["mite"] is not None


@pytest.mark.skip(reason="DistRegBuilder is currently broken.")
class TestDistRegBuilderMCMC:
    @pytest.mark.mcmc
    def test_lin_mcmc(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_linear("x1", "x2", m=0.0, s=100.0, param="loc")
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()
        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        scale = model.vars["scale"]
        eb.positions_included = [scale.name]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        b0 = samples["loc_intercept"].mean().round()
        b = np.squeeze(samples["loc_linear0_coef"].mean(axis=1).round())
        g0 = samples["scale"].mean().round(1)

        assert b0 == pytest.approx(1.0)
        assert b == pytest.approx(np.array([0.0, 2.0]))
        assert g0 == pytest.approx(2.0, abs=0.5)

    @pytest.mark.mcmc
    def test_ps_mcmc(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_pspline("x1", nparam=7, a=2.0, b=0.5, param="loc")
            .add_pspline("x2", nparam=7, a=2.0, b=0.5, param="loc")
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()

        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        loc = model.vars["loc"]
        scale = model.vars["scale"]
        eb.positions_included = [loc.name, scale.name]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        predicted_values = np.squeeze(samples["loc"].mean(axis=1))
        diff = y - predicted_values
        assert np.sum(diff) == pytest.approx(0.0, abs=3.0)

        g0 = samples["scale"].mean().round(1)
        assert g0 == pytest.approx(2.0, abs=0.5)

    @pytest.mark.mcmc
    def test_te_mcmc(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_teprod_full("x1", "x2", nparam=(7, 7), a=2.0, b=0.5, param="loc")
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()

        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        loc = model.vars["loc"]
        scale = model.vars["scale"]
        eb.positions_included = [loc.name, scale.name]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        predicted_values = np.squeeze(samples["loc"].mean(axis=1))
        diff = y - predicted_values
        assert np.sum(diff) == pytest.approx(0.0, abs=3.0)

        g0 = samples["scale"].mean().round(1)
        assert g0 == pytest.approx(2.0, abs=0.5)

    @pytest.mark.mcmc
    def test_ti_mcmc(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_intercept(param="loc")
            .add_pspline("x1", nparam=7, a=2.0, b=0.5, param="loc")
            .add_pspline("x2", nparam=7, a=2.0, b=0.5, param="loc")
            .add_teprod_interaction(
                "x1", "x2", nparam=(7, 7), a=2.0, b=0.5, param="loc"
            )
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()

        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        loc = model.vars["loc"]
        scale = model.vars["scale"]
        eb.positions_included = [loc.name, scale.name]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        predicted_values = np.squeeze(samples["loc"].mean(axis=1))
        diff = y - predicted_values
        assert np.sum(diff) == pytest.approx(0.0, abs=3.0)

        g0 = samples["scale"].mean().round(1)
        assert g0 == pytest.approx(2.0, abs=0.5)

    @pytest.mark.mcmc
    def test_mips_mcmc(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_pspline_mi("x1", nparam=7, a=2.0, b=0.5, param="loc", name="mi")
            .add_pspline("x2", nparam=7, a=2.0, b=0.5, param="loc")
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()

        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        loc = model.vars["loc"]
        scale = model.vars["scale"]
        eb.positions_included = [loc.name, scale.name, "mi_positive_coef"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        X = model.groups()["mi"].X.value
        mean_coef = np.squeeze(samples["mi_positive_coef"].mean(axis=1))
        mi_smooth = X @ mean_coef
        diff = np.diff(mi_smooth)
        assert np.all(diff >= 0.0)

        predicted_values = np.squeeze(samples["loc"].mean(axis=1))
        diff = y - predicted_values
        assert np.sum(diff) == pytest.approx(0.0, abs=3.0)

        g0 = samples["scale"].mean().round(1)
        assert g0 == pytest.approx(2.0, abs=1)

    @pytest.mark.mcmc
    def test_mite1_mcmc(self, drb: DistRegBuilder) -> None:
        drb = (
            drb.add_teprod_mi1_full(
                "x1", "x2", nparam=(7, 7), a=2.0, b=0.5, param="loc", name="mite"
            )
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()

        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        loc = model.vars["loc"]
        scale = model.vars["scale"]
        eb.positions_included = [loc.name, scale.name, "mite_positive_coef"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        mean_coef = np.squeeze(samples["mite_positive_coef"].mean(axis=1))
        mean_coef = np.insert(mean_coef, 0, 0)

        coef_matrix = np.reshape(mean_coef, (7, 7))
        assert np.all(coef_matrix[1:, :] >= 0)

        predicted_values = np.squeeze(samples["loc"].mean(axis=1))
        diff = y - predicted_values
        assert np.sum(diff) == pytest.approx(0.0, abs=3.0)

        g0 = samples["scale"].mean().round(1)
        assert g0 == pytest.approx(2.0, abs=0.5)

    @pytest.mark.mcmc
    def test_mite1_mcmc2(self, drb: DistRegBuilder) -> None:
        """Same as above, but this time monotone in x2 instead of x1."""
        drb = (
            drb.add_teprod_mi1_full(
                "x2", "x1", nparam=(7, 7), a=2.0, b=0.5, param="loc", name="mite"
            )
            .add_intercept(param="loc")
            .add_intercept(param="scale")
            .add_param("loc", tfb.Identity)
            .add_param("scale", tfb.Exp)
            .add_response("y", tfd.Normal)
        )

        model = drb.build_model()

        eb = dist_reg_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        loc = model.vars["loc"]
        scale = model.vars["scale"]
        eb.positions_included = [loc.name, scale.name, "mite_positive_coef"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        mean_coef = np.squeeze(samples["mite_positive_coef"].mean(axis=1))
        mean_coef = np.insert(mean_coef, 0, 0)

        coef_matrix = np.reshape(mean_coef, (7, 7))
        assert np.all(coef_matrix[1:, :] >= 0)

        predicted_values = np.squeeze(samples["loc"].mean(axis=1))
        diff = y - predicted_values
        assert np.sum(diff) == pytest.approx(0.0, abs=3.0)

        g0 = samples["scale"].mean().round(1)
        assert g0 == pytest.approx(2.0, abs=0.5)
