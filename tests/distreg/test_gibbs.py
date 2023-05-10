from typing import Iterator

import jax
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pytest

import liesel_bctm.distreg.gibbs as gb
import liesel_bctm.distreg.psplines as ps

x = np.random.uniform(size=20)


@pytest.fixture
def tp() -> Iterator[ps.PSplineTP]:
    yield ps.PSplineTP("test", x=(x, x), nparam=(7, 7), a=2.0, b=0.5)


@pytest.fixture
def pspline() -> Iterator[ps.PSpline]:
    yield ps.PSpline("test", x, nparam=7, a=2.0, b=0.5)


@pytest.fixture
def model(tp: ps.PSplineTP) -> Iterator[lsl.Model]:
    gb = lsl.GraphBuilder().add(tp.smooth)
    model = gb.build_model()
    yield model


class TestVarGibbsKernel:
    def test_generate_kernel(self, pspline: ps.PSpline) -> None:
        kernel = gb.igvar_gibbs_kernel(pspline)
        assert kernel is not None

    def test_transition_ps(self, pspline: ps.PSpline) -> None:
        key = jax.random.PRNGKey(1)
        model = lsl.GraphBuilder().add(pspline.smooth).build_model()
        kernel = gb.igvar_gibbs_kernel(pspline)
        draw = kernel._transition_fn(key, model_state=model.state)
        assert draw["test_igvar"].round(2) == pytest.approx(0.16)

    def test_transition_ps_jit(self, pspline: ps.PSpline) -> None:
        key = jax.random.PRNGKey(1)
        model = lsl.GraphBuilder().add(pspline.smooth).build_model()
        kernel = gb.igvar_gibbs_kernel(pspline)
        draw = jax.jit(kernel._transition_fn)(key, model_state=model.state)
        assert draw["test_igvar"].round(2) == pytest.approx(0.16)

    def test_transition_tp(self, tp: ps.PSplineTP, model: lsl.Model) -> None:
        key = jax.random.PRNGKey(1)

        kernel = gb.igvar_gibbs_kernel(tp)
        draw = kernel._transition_fn(key, model_state=model.state)
        assert draw["test_igvar"].round(2) == pytest.approx(0.03)

    def test_transition_tp_jit(self, tp: ps.PSplineTP, model: lsl.Model) -> None:
        key = jax.random.PRNGKey(1)

        kernel = gb.igvar_gibbs_kernel(tp)
        draw = jax.jit(kernel._transition_fn)(key, model_state=model.state)
        assert draw["test_igvar"].round(2) == pytest.approx(0.03)


class TestWeightGibbsKernel:
    def test_generate_weight_kernel(self, tp: ps.PSplineTP) -> None:
        kernel = gb.weight_gibbs_kernel(tp)
        assert isinstance(kernel, gs.GibbsKernel)

    def test_coef_log_probs(self, tp: ps.PSplineTP, model: lsl.Model) -> None:
        log_probs = gb._coef_log_probs(tp, model.state)
        assert not np.any(np.isnan(log_probs))

    def test_coef_log_probs_jit(self, tp: ps.PSplineTP, model: lsl.Model) -> None:
        fn = jax.jit(gb._coef_log_probs, static_argnames=["group"])
        log_probs = fn(tp, model.state)
        assert not np.any(np.isnan(log_probs))

    def test_run_weight_kernel(self, tp: ps.PSplineTP, model: lsl.Model) -> None:
        key = jax.random.PRNGKey(1)

        kernel = gb.weight_gibbs_kernel(tp)
        draw = kernel._transition_fn(key, model_state=model.state)
        assert draw["test_penalty_weight"].round(2) == pytest.approx(0.36)

    def test_run_weight_kernel_jit(self, tp: ps.PSplineTP, model: lsl.Model) -> None:
        key = jax.random.PRNGKey(1)

        kernel = gb.weight_gibbs_kernel(tp)
        draw = jax.jit(kernel._transition_fn)(key, model_state=model.state)
        assert draw["test_penalty_weight"].round(2) == pytest.approx(0.36)
