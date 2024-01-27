import jax.numpy as jnp
from liesel_bctm.custom_types import Array
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.model import Calc, Dist
from liesel.model import Group as LieselGroup
from liesel.model import Obs, Param, Var

from ..custom_types import Array, Kernel


class Group(LieselGroup):

    return_kernels: bool = False
    """If False, the group's :meth:`.gibbs_kernels` method returns an empty list."""

    sampled_params: list[str] = []
    """Names of the params that can be sampled with samplers like NUTS or IWLS."""

    def gibbs_kernels(self) -> list[Kernel]:
        if not self.return_kernels:
            return []

        return self._gibbs_kernels()


class SVar(Var):
    """
    Standardised variable. Requires an array input, not a calculator.
    """

    def __init__(self, value: Array, distribution: Dist | None = None, name: str = ""):
        self.mean = np.mean(value, dtype=np.float32)
        """Variable mean."""

        self.scale = np.std(value, dtype=np.float32)
        """Variable scale."""

        value_standardized = np.array(
            (value - self.mean) / self.scale, dtype=np.float32
        )
        super().__init__(value_standardized, distribution, name)

    def rescale(self, value: Array) -> Array:
        """
        Rescales the input value by multiplying it with the saved scale and adding the
        saved mean.
        """
        return (value * self.scale) + self.mean


class Intercept(Group):
    def __init__(self, name: str) -> None:
        self.intercept = Param(np.zeros(1, dtype=np.float32), name=name)
        """The intercept parameter."""

        self.sampled_params = [self.intercept.name]

        super().__init__(name, intercept=self.intercept)


class Lin(Group):
    """
    Represents a linear smooth with a normal prior for the regression
    coefficient.

    Parameters
    ----------
    name
        Group name.
    x
        Design matrix.
    m
        Mean for the normal prior of the smooth coefficient.
    s
        Scale for the normal prior of the smooth coefficient.
    """

    def __init__(self, name: str, x: Array, m: float, s: float) -> None:
        self.x = Obs(np.atleast_2d(x).astype(np.float32), name=name + "_x")
        """The design matrix for this smooth."""

        self.m = Var(m, name=name + "_m")
        """The prior mean for this smooth's coefficients."""

        self.s = Var(s, name=name + "_s")
        """The prior scale for this smooth's coefficients."""

        nparam = np.shape(x)[-1]
        init = np.zeros(nparam, dtype=np.float32)
        prior = Dist(tfd.Normal, loc=self.m, scale=self.s)

        self.coef = Param(init, prior, name + "_coef")
        """The smooth's regression coef."""

        self.smooth = Var(Calc(jnp.dot, self.x, self.coef), name=name)
        """The smooth ``x @ coef``."""

        self.sampled_params = [self.coef.name]

        super().__init__(
            name, x=self.x, m=self.m, s=self.s, coef=self.coef, smooth=self.smooth
        )

    def ppeval(self, samples: dict, x: Array | None = None) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        coef_samples = samples[self.coef.name]
        if x is None:
            X = self.x.value
        elif len(np.shape(x)) >= 2:
            X = x
        elif len(np.shape(x)) <= 1:
            X = np.atleast_2d(x).T
        smooth = np.tensordot(X, coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)


class LinConst(Group):
    """
    Represents a linear smooth with a constant prior for the regression
    coefficient.

    Parameters
    ----------
    name
        Group name.
    x
        Design matrix.
    """

    def __init__(self, name: str, x: Array) -> None:
        self.x = Obs(np.atleast_2d(x).astype(np.float32), name=name + "_x")
        """The design matrix for this smooth."""

        nparam = np.shape(x)[-1]
        init = np.zeros(nparam, dtype=np.float32)

        self.coef = Param(init, name=name + "_coef")
        """The smooth's regression coef."""

        self.smooth = Var(Calc(jnp.dot, self.x, self.coef), name=name)
        """The smooth ``x @ coef``."""

        self.sampled_params = [self.coef.name]

        super().__init__(
            name, x=self.x, coef=self.coef, smooth=self.smooth
        )

    def ppeval(self, samples: dict, x: Array | None = None) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        coef_samples = samples[self.coef.name]
        if x is None:
            X = self.x.value
        elif len(np.shape(x)) >= 2:
            X = x
        elif len(np.shape(x)) <= 1:
            X = np.atleast_2d(x).T
        smooth = np.tensordot(X, coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)

