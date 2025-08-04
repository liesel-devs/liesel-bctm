from typing import Iterator

import numpy as np
import pytest
import scipy

from liesel_bctm.builder import CTMBuilder, MIPSDerivative, MITEDerivative, ctm_mcmc
from liesel_bctm.custom_types import Array
from liesel_bctm.distreg import mi_splines as mi

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
def ctmb() -> Iterator[CTMBuilder]:
    yield CTMBuilder(data)


class TestMIPSplinePosteriorPred:
    def test_ppeval(self):
        mipspline = mi.MIPSpline("test", x=x[:, 0], nparam=7, a=2.0, b=0.5)
        mipspline = MIPSDerivative(mipspline)
        # shape: (nchains, nsamples, nparam)
        coef_samples = np.exp(np.random.normal(size=(3, 10, 6)))
        # 1 param less because of identifiability constraint
        samples = {mipspline.pt.coef.name: coef_samples}

        prediction = mipspline.ppeval(samples, x=0.5)
        assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

        prediction = mipspline.ppeval(samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)
        assert np.all((prediction + 1e-06) >= 0.0)

        aggregated_samples = {mipspline.pt.coef.name: coef_samples.mean(axis=1)}
        # (nchains, nparam)
        assert aggregated_samples[mipspline.pt.coef.name].shape == (
            3,
            6,
        )

        prediction = mipspline.ppeval(aggregated_samples, x=np.array([0.3, 0.6]))
        assert prediction.shape == (3, 2)  # (nchains, nsamples, nvalues)


@pytest.fixture
def tp() -> Iterator[MITEDerivative]:
    yield MITEDerivative(
        mi.MIPSplineTE1("test", x=(x[:, 0], x[:, 1]), nparam=(7, 7), a=2.0, b=0.5)
    )


@pytest.fixture
def samples(tp: MITEDerivative) -> Iterator[dict[str, Array]]:
    coef_samples = np.random.normal(
        size=(3, 10, 48)
    )  # shape: (nchains, nsamples, nparam)
    # 1 param less because of identifiability constraint
    samples = {tp.pt.coef.name: np.exp(coef_samples)}
    yield samples


class TestMIPSplineTE1PosteriorPred:
    def test_ppeval_both_scalars(
        self, tp: MITEDerivative, samples: dict[str, Array]
    ) -> None:
        prediction = tp.ppeval(samples, x=(0.5, 0.4))
        assert prediction.shape == (3, 10, 1)  # (nchains, nsamples, nvalues)

    def test_ppeval_both_arrays(
        self, tp: MITEDerivative, samples: dict[str, Array]
    ) -> None:
        a = np.array([0.3, 0.6])
        b = np.array([0.2, 0.8])
        prediction = tp.ppeval(samples, x=(a, b))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

    def test_ppeval_one_scalar(
        self, tp: MITEDerivative, samples: dict[str, Array]
    ) -> None:
        a = np.array([0.3, 0.6])
        b = 0.2
        prediction = tp.ppeval(samples, x=(a, b))
        assert prediction.shape == (3, 10, 2)  # (nchains, nsamples, nvalues)

    def test_ppeval_both_arrays_different_length(
        self, tp: MITEDerivative, samples: dict[str, Array]
    ) -> None:
        with pytest.raises(ValueError, match="equal shape"):
            a = np.array([0.3, 0.6])
            b = np.array([0.2, 0.8, 0.7])
            tp.ppeval(samples, x=(a, b))


class TestCTMBuilder:
    def test_init(self, ctmb: CTMBuilder) -> None:
        assert ctmb is not None

    def test_rmips(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_lin(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_linear("x1", "x2", m=0.0, s=100.0)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_ps(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_pspline("x1", nparam=7, a=2.0, b=0.5)
            .add_pspline("x2", nparam=7, a=2.0, b=0.5)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_te(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_teprod_full("x1", "x2", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_ti(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_teprod_interaction("x1", "x2", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_mips(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_pspline_mi("x1", nparam=7, a=2.0, b=0.5)
            .add_pspline("x2", nparam=7, a=2.0, b=0.5)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_mite1(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_teprod_mi1_full("x1", "x2", nparam=(7, 7), a=2.0, b=0.5, name="mite")
            .add_pspline("x2", nparam=7, a=2.0, b=0.5)
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)

    def test_trafo_teprod_full(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo_teprod_full("y", "x2", nparam=(7, 7), a=2.0, b=0.5, name="mite")
            .add_response("y")
        )
        model = ctmb.build_model()
        assert (model.log_prior + model.log_lik) == pytest.approx(model.log_prob)


class TestDistRegBuilderMCMC:
    @pytest.mark.mcmc
    def test_trafo_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_intercept()
            .add_response("y")
        )

        model = ctmb.build_model()
        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p > 0.05

    @pytest.mark.mcmc
    def test_trafo_mcmc_uniform(self) -> None:
        """Test successfull normalisation of a uniform response."""
        y = rng.uniform(low=-2, high=2, size=n)
        ctmb = (
            CTMBuilder()
            .add_trafo(y, nparam=7, a=2.0, b=0.5)
            .add_intercept()
            .add_response(y)
        )

        model = ctmb.build_model()
        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p > 0.05

    @pytest.mark.mcmc
    def test_lin_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_linear("x1", "x2", m=0.0, s=100.0)
            .add_response("y")
        )

        model = ctmb.build_model()
        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p > 0.05

        b = np.squeeze(samples["linear0_coef"].mean(axis=1).round())
        assert b == pytest.approx(np.array([0.0, -1.0]))

    @pytest.mark.mcmc
    def test_ps_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_pspline("x1", nparam=7, a=2.0, b=0.5)
            .add_pspline("x2", nparam=7, a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p > 0.05

    @pytest.mark.mcmc
    def test_te_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_teprod_full("x1", "x2", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p >= 0.05

    @pytest.mark.mcmc
    def test_ti_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_teprod_interaction("x1", "x2", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p >= 0.05

    @pytest.mark.mcmc
    def test_mips_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_pspline_mi("x1", nparam=7, a=2.0, b=0.5)
            .add_pspline("x2", nparam=7, a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        # include predicted values
        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p > 0.05

    @pytest.mark.mcmc
    def test_mite1_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo("y", nparam=7, a=2.0, b=0.5)
            .add_teprod_mi1_full("x1", "x2", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p >= 0.05

    # @pytest.mark.mcmc
    def test_trafo_teprod_full_mcmc(self) -> None:
        ctmb = (
            CTMBuilder(data)
            .add_intercept()
            .add_trafo_teprod_full("y", "x2", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p >= 0.05

    @pytest.mark.mcmc
    def test_trafo_teprod_full_mcmc2(self) -> None:
        """
        This test is a bit more challenging, since the response scale also depends on
        the covariate.
        """
        x = rng.uniform(-2, 2, size=n)
        y = rng.normal(loc=2.0 + np.cos(x), scale=np.sqrt(np.exp(x)))
        ctmb = (
            CTMBuilder(data={"y": y, "x": x})
            .add_intercept()
            .add_trafo_teprod_full("y", "x", nparam=(7, 7), a=2.0, b=0.5)
            .add_response("y")
        )

        model = ctmb.build_model()

        eb = ctm_mcmc(model, seed=1, num_chains=1)
        eb.set_duration(warmup_duration=500, posterior_duration=1000)

        eb.positions_included = ["z"]

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        z = np.squeeze(samples["z"].mean(axis=1))
        assert z.shape == (n,)
        assert z.mean() == pytest.approx(0.0, abs=0.2)
        assert z.std() == pytest.approx(1.0, abs=0.2)

        _, p = scipy.stats.normaltest(z)
        assert p >= 0.05
