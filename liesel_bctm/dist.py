from __future__ import annotations

import jax.numpy as jnp
from liesel.model import Dist, Var
from liesel.model.nodes import no_model_setter

from .custom_types import Array, TFPDistribution


class TDist(Dist):
    """A transformation-distribution node for a conditional transformation model."""

    def __init__(
        self,
        ct: Var,
        ctd: Var,
        refdist: TFPDistribution,
        _name: str = "",
        _needs_seed: bool = False,
    ):
        super(Dist, self).__init__(ct, ctd, _name=_name, _needs_seed=_needs_seed)

        self._per_obs = True
        self.refdist = refdist
        self._ct = ct
        self._ctd = ctd

    @property
    def log_prob(self) -> Array:
        """The log-probability of the distribution."""
        return self.value

    @property
    def per_obs(self) -> bool:
        """Whether the log-probability is stored per observation or summed up."""
        return self._per_obs

    @per_obs.setter
    @no_model_setter
    def per_obs(self, per_obs: bool):
        self._per_obs = per_obs

    def update(self) -> TDist:
        base_log_prob = self.refdist.log_prob(self._ct.value)
        log_prob_adjustment = jnp.log(self._ctd.value)
        log_prob = jnp.add(base_log_prob, log_prob_adjustment)

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()

        self._value = log_prob
        self._outdated = False
        return self
