import numpy as np

from liesel_bctm.builder import CTMBuilder, ctm_mcmc

rng = np.random.default_rng(seed=1)

n = 100
x = rng.uniform(-2, 2, size=(n, 2))
X = np.column_stack((np.ones(n), np.sort(x[:, 0]), x[:, 1]))
coef = np.array([1.0, 0.0, 2.0])
y = rng.normal(loc=X @ coef, scale=2)
data = {"y": y, "x1": np.sort(x[:, 0]), "x2": x[:, 1]}

ctmb = (
    CTMBuilder(data)
    .add_intercept()
    .add_trafo_teprod_full(
        "y", "x2", nparam=(7, 7), a=2.0, b=0.5, name="trafo_teprod_full"
    )
    .add_linear("x1", m=0.0, s=100.0, name="x1")
    .add_response("y")
)


def run(builder):
    model = builder.build_model()

    eb = ctm_mcmc(model, seed=1, num_chains=2)
    eb.set_duration(warmup_duration=500, posterior_duration=1000)

    eb.positions_included = [
        "z",
        "trafo_teprod_full",
        "x1",
        "trafo_teprod_full_positive_coef",
    ]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    return results


if __name__ == "__main__":
    results = run(ctmb)
    results.pkl_save("tests/files/results.pickle")
