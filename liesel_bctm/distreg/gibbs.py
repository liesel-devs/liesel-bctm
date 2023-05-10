from __future__ import annotations

import jax.numpy as jnp
import jax.random
from liesel.distributions import MultivariateNormalDegenerate
from liesel.goose import GibbsKernel
from liesel.model.nodes import Array, Group, NodeState

from ..custom_types import KeyArray


def igvar_gibbs_kernel(group: Group) -> GibbsKernel:
    """
    Builds a Gibbs kernel for a smoothing parameter with an inverse gamma prior.

    The group must have the following members:

    - var_param: The variance parameter
    - a: Shape parameter of the IG prior for var
    - b: Rate parameter of the IG prior for var
    - rank: Rank of the penalty matrix
    - penalty: Penalty matrix
    - coef: Spline coefficient
    """
    position_key = group["var_param"].name

    def transition(prng_key, model_state: dict[str, NodeState]) -> dict[str, Array]:
        a = group.value_from(model_state, "a")
        rank = group.value_from(model_state, "rank")

        a_gibbs = jnp.squeeze(a + 0.5 * rank)

        b = group.value_from(model_state, "b")
        coef = group.value_from(model_state, "coef")
        pen = group.value_from(model_state, "penalty")

        b_gibbs = jnp.squeeze(b + 0.5 * (coef @ pen @ coef))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {position_key: draw}

    return GibbsKernel([position_key], transition)


def weight_gibbs_kernel(group: Group) -> GibbsKernel:
    position_key = group["weight"].name
    weight_grid = group["weight_grid"].value

    def transition(
        prng_key: KeyArray, model_state: dict[str, NodeState]
    ) -> dict[str, Array]:
        log_probs = _coef_log_probs(group, model_state)
        draw = jax.random.categorical(prng_key, logits=log_probs)

        return {position_key: weight_grid[draw]}

    return GibbsKernel([position_key], transition)


def _coef_log_probs(group, model_state: dict[str, NodeState]) -> Array:
    """Helper for defining the weights Gibbs kernel."""
    weight_grid = group["weight_grid"].value

    coef = group.value_from(model_state, "coef")
    var = group.value_from(model_state, "var_param")
    m = model_state[group["coef"].m.value_node.name].value

    def coef_logprob(weight: float) -> Array:
        penalty = group["penalty"].value_node.function(weight)
        rank = group["rank"].value
        log_pdet = group["log_pdet"].lookup_calculator.function(weight)
        mvn = MultivariateNormalDegenerate.from_penalty(
            loc=m, var=var, pen=penalty, rank=rank, log_pdet=log_pdet
        )
        return mvn.log_prob(coef)

    log_probs = jnp.squeeze(jax.vmap(coef_logprob)(weight_grid))
    return log_probs
