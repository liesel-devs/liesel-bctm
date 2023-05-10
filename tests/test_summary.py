from typing import Iterator

import liesel.goose as gs
import numpy as np
import pytest

from liesel_bctm.custom_types import Array
from liesel_bctm.summary import (
    ConditionalPredictions,
    cdist_quantiles,
    cdist_quantiles_early,
    cdist_rsample,
    grid,
    partial_ctrans_df_early,
    partial_ctrans_quantiles,
    partial_ctrans_rsample,
    sample_dgf,
    sample_quantiles,
)

from .files.run_model import ctmb

# components: "trafo_teprod_full", "x1", "intercept"


@pytest.fixture(scope="module")
def samples() -> Iterator[dict[str, Array]]:
    results = gs.engine.SamplingResults.pkl_load("tests/files/results.pickle")
    yield results.get_posterior_samples()


class TestConditionalPredictionsLantentVariable:
    def test_intercept(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb)

        intercept = ctmp.intercept()
        intercept_samples = samples["intercept"]
        assert np.allclose(intercept, intercept_samples)

    def test_empty(self, samples: dict[str, Array]) -> None:
        """
        Test depends on tests/files/results.pickle. If it fails after an update to the
        package, it might help to re-generate that file by running
        tests/files/run_model.py.
        """
        ctmp = ConditionalPredictions(samples, ctmb)
        z = ctmp.ctrans()
        assert np.allclose(z, samples["z"], atol=0.1)

    def test_fix(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=(np.linspace(0, 1, 10), 0.0), x1=0.0
        )
        z = ctmp.ctrans()
        assert z.shape == (2, 1000, 10)
        assert not np.all(z == 0.0)

        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=(None, 0.0), x1=0.0
        )
        z = ctmp.ctrans()
        assert z.shape == (2, 1000, 100)
        assert not np.all(z == 0.0)

    def test_grid(self, samples: dict[str, Array]) -> None:
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)

        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=grid(y, x2), x1=0.3
        )
        z = ctmp.ctrans()
        assert z.shape == (2, 1000, 15 * 3)

    def test_empty_d(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb)
        zd = ctmp.ctrans_d()
        assert zd.shape == (2, 1000, 100)

    def test_fix_d(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=(np.linspace(0, 1, 10), 0.0), x1=0.0
        )
        z1 = ctmp.ctrans_d()
        assert z1.shape == (2, 1000, 10)
        assert not np.all(z1 == 0.0)

        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=(np.linspace(0, 1, 10), 0.0), x1=3.0
        )
        z2 = ctmp.ctrans_d()
        assert np.allclose(z1, z2)


class TestConditionalPredictionsDist:
    def test_log_prob_empty(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb)
        lp = ctmp.log_prob()
        assert lp.shape == samples["z"].shape

    def test_log_prob_fix(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=(None, 0.0), x1=0.0
        )
        lp = ctmp.log_prob()

        assert lp.shape == (2, 1000, 100)

    def test_log_prob_grid(self, samples: dict[str, Array]) -> None:
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)

        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=grid(y, x2), x1=0.3
        )
        lp = ctmp.log_prob()
        assert lp.shape == (2, 1000, 15 * 3)

    def test_pdf(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb)
        assert ctmp.pdf().shape == samples["z"].shape

    def test_cdf(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb)
        cdf = ctmp.cdf()
        assert cdf.shape == samples["z"].shape
        assert np.all(cdf >= 0)
        assert np.all(cdf <= 1)


class TestPartialConditionalPredictions:
    def test_empty(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb)
        with pytest.raises(ValueError, match="No smooth specified"):
            ctmp.partial_ctrans()

    def test_partial_ctrans_trafo_teprod_full(self, samples: dict[str, Array]) -> None:
        """
        Test depends on tests/files/results.pickle. If it fails after an update to the
        package, it might help to re-generate that file by running
        tests/files/run_model.py.
        """
        ctmp = ConditionalPredictions(samples, ctmb, trafo_teprod_full=None)
        trafo_teprod_full = ctmp.partial_ctrans()
        trafo_teprod_full_samples = samples["trafo_teprod_full"]
        assert trafo_teprod_full.shape == trafo_teprod_full_samples.shape
        assert np.allclose(trafo_teprod_full, trafo_teprod_full_samples, atol=0.1)

    def test_partial_ctrans_x1(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb, x1=None)
        x1 = ctmp.partial_ctrans()
        x1_samples = samples["x1"]
        assert x1.shape == x1_samples.shape
        assert np.allclose(x1, x1_samples, atol=0.1)

    def test_partial_ctrans_trafo_teprod_full_and_x1(
        self, samples: dict[str, Array]
    ) -> None:
        """
        Test depends on tests/files/results.pickle. If it fails after an update to the
        package, it might help to re-generate that file by running
        tests/files/run_model.py.
        """
        ctmp = ConditionalPredictions(samples, ctmb, trafo_teprod_full=None, x1=None)
        z = ctmp.partial_ctrans()
        trafo_teprod_full_samples = samples["trafo_teprod_full"]
        x1_samples = samples["x1"]
        z_samples = trafo_teprod_full_samples + x1_samples

        assert z.shape == z_samples.shape
        assert np.allclose(z, z_samples, atol=0.1)

    def test_partial_ctrans_trafo_teprod_full_and_x1_fixed(
        self, samples: dict[str, Array]
    ) -> None:

        with pytest.raises(ValueError, match="must be of equal shape or scalar"):
            ctmp = ConditionalPredictions(
                samples, ctmb, trafo_teprod_full=(None, np.linspace(0, 1, 7)), x1=0.3
            )
            ctmp.partial_ctrans()

        # how to do it with a grid
        y = np.linspace(0, 1, 10)
        x2 = np.linspace(0, 1, 3)
        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=grid(y, x2), x1=0.3
        )
        z = ctmp.partial_ctrans()

        assert z.shape == (2, 1000, 30)

        # fails for shape mismatch
        with pytest.raises(ValueError):
            ctmp = ConditionalPredictions(
                samples, ctmb, trafo_teprod_full=grid(y, x2), x1=np.array([0.3, 0.4])
            )
            ctmp.partial_ctrans()

    def test_partial_ctrans_d(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb, x1=None)

        x1 = ctmp.partial_ctrans_d()
        assert x1 == pytest.approx(0.0)

        ctmp = ConditionalPredictions(samples, ctmb, trafo_teprod_full=None)
        trafo_teprod_full = ctmp.partial_ctrans_d()
        assert np.all(trafo_teprod_full >= 0)
        assert trafo_teprod_full.shape == (2, 1000, 100)

    def test_partial_ctrans_d_fixed(self, samples: dict[str, Array]) -> None:
        ctmp = ConditionalPredictions(samples, ctmb, trafo_teprod_full=(None, 1.5))
        trafo_teprod_full = ctmp.partial_ctrans_d()
        assert np.all(trafo_teprod_full >= 0)
        assert trafo_teprod_full.shape == (2, 1000, 100)

        ctmp = ConditionalPredictions(
            samples, ctmb, trafo_teprod_full=(np.linspace(0, 0.5, 2), 1.5)
        )
        trafo_teprod_full = ctmp.partial_ctrans_d()
        assert np.all(trafo_teprod_full >= 0)
        assert trafo_teprod_full.shape == (2, 1000, 2)


class TestSummaryDataFrames:
    def test_summary_helper(self, samples: dict[str, Array]) -> None:

        q01 = sample_quantiles(samples, q=0.1)
        q50 = sample_quantiles(samples, q=0.5)
        q90 = sample_quantiles(samples, q=0.9)

        lp01 = ConditionalPredictions(q01, ctmb).log_prob()
        lp50 = ConditionalPredictions(q50, ctmb).log_prob()
        lp90 = ConditionalPredictions(q90, ctmb).log_prob()

        assert lp01.shape == lp50.shape
        assert lp50.shape == lp90.shape

    def test_summarise_dist(self, samples: dict[str, Array]) -> None:
        """This test only ensures that the function is running."""
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)
        x1 = 0.3
        df = cdist_quantiles(samples, ctmb, trafo_teprod_full=grid(y, x2), x1=x1)
        assert df.shape == (180, 8)

    def test_partial_ctrans_df(self, samples: dict[str, Array]) -> None:
        """This test only ensures that the function is running."""
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)
        x1 = 0.3
        df = partial_ctrans_quantiles(
            samples, ctmb, trafo_teprod_full=grid(y, x2), x1=x1
        )
        assert df.shape == (180, 7)

    def test_partial_ctrans_df_only_cov(self, samples: dict[str, Array]) -> None:
        """This test only ensures that the function is running."""
        x1 = np.linspace(0, 1, 5).reshape((5, 1))
        df = partial_ctrans_quantiles(samples, ctmb, x1=x1)
        assert df.shape == (20, 4)

    def test_summarise_dist_early(self, samples: dict[str, Array]) -> None:
        """This test only ensures that the function is running."""
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)
        x1 = 0.3
        df = cdist_quantiles_early(samples, ctmb, trafo_teprod_full=grid(y, x2), x1=x1)
        assert df is not None

    def test_partial_ctrans_df_early(self, samples: dict[str, Array]) -> None:
        """This test only ensures that the function is running."""
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)
        df = partial_ctrans_df_early(samples, ctmb, trafo_teprod_full=grid(y, x2))
        assert df is not None

    def test_sample_grid(self) -> None:
        def dgf(x1, x2, x3, n):
            return np.random.normal(loc=x1 + x2 - x3, size=n)

        df = sample_dgf(dgf, n=10, x1=[1, 2, 3], x2=np.array([0.4]), x3=[8.0])
        assert df.shape == (30, 5)
        assert list(df.columns.values) == ["index", "x1", "x2", "x3", "y"]

    def test_rdist(self, samples: dict[str, Array]) -> None:
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)
        x1 = 0.3

        df = cdist_rsample(samples, ctmb, size=5, trafo_teprod_full=grid(y, x2), x1=x1)
        assert df.shape == (5 * 45, 7)

        df = cdist_rsample(samples, ctmb, size=10, trafo_teprod_full=grid(y, x2), x1=x1)
        assert df.shape == (10 * 45, 7)

    def test_partial_ctrans_rsample(self, samples: dict[str, Array]) -> None:
        y = np.linspace(0, 1, 15)
        x2 = np.linspace(0, 1, 3)
        df = partial_ctrans_rsample(
            samples, ctmb, size=10, trafo_teprod_full=(grid(y, x2))
        )
        assert df.shape == (10 * 45, 5)
