import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.distributions import MultivariateNormalDegenerate
from liesel.goose import GibbsKernel, NUTSKernel
from liesel.model import Group as LieselGroup
from liesel.model import Node, Var
from liesel.model.nodes import Calc, Data, Dist
from sklearn.preprocessing import LabelBinarizer

from ..custom_types import Array, Kernel


class Group(LieselGroup):
    return_kernels: bool = False
    """If False, the group's :meth:`.gibbs_kernels` method returns an empty list."""

    sampled_params: list[str] = []
    """Names of the params that can be sampled with samplers like NUTS or IWLS."""

    smooth: Var

    def _gibbs_kernels(self) -> list[GibbsKernel]:
        raise NotImplementedError

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
        self.intercept = Var.new_param(np.zeros(1, dtype=np.float32), name=name)
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
        self.x = Var.new_obs(np.atleast_2d(x).astype(np.float32), name=name + "_x")
        """The design matrix for this smooth."""

        self.m = Var(m, name=name + "_m")
        """The prior mean for this smooth's coefficients."""

        self.s = Var(s, name=name + "_s")
        """The prior scale for this smooth's coefficients."""

        nparam = np.shape(x)[-1]
        init = np.zeros(nparam, dtype=np.float32)
        prior = Dist(tfd.Normal, loc=self.m, scale=self.s)

        self.coef = Var.new_param(init, prior, name + "_coef")
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
        elif len(jnp.shape(x)) >= 2:
            X = x
        elif len(jnp.shape(x)) <= 1:
            X = jnp.atleast_2d(x).T
        smooth = jnp.tensordot(X, coef_samples, axes=([1], [-1]))
        return jnp.moveaxis(smooth, 0, -1)


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
        self.x = Var.new_obs(np.atleast_2d(x).astype(np.float32), name=name + "_x")
        """The design matrix for this smooth."""

        nparam = np.shape(x)[-1]
        init = np.zeros(nparam, dtype=np.float32)

        self.coef = Var.new_param(init, name=name + "_coef")
        """The smooth's regression coef."""

        self.smooth = Var(Calc(jnp.dot, self.x, self.coef), name=name)
        """The smooth ``x @ coef``."""

        self.sampled_params = [self.coef.name]

        super().__init__(name, x=self.x, coef=self.coef, smooth=self.smooth)

    def ppeval(self, samples: dict, x: Array | None = None) -> Array:
        """
        Return posterior predictive evaluation of this smooth given an array of samples.

        Uses :attr:`.X.value` if ``x=None``.
        """
        coef_samples = samples[self.coef.name]
        if x is None:
            X = self.x.value
        elif len(jnp.shape(x)) >= 2:
            X = x
        elif len(jnp.shape(x)) <= 1:
            X = jnp.atleast_2d(x).T
        smooth = jnp.tensordot(X, coef_samples, axes=([1], [-1]))
        return jnp.moveaxis(smooth, 0, -1)


def find_param(var: Var) -> Var | None:
    if var.parameter:
        if not var.strong:
            raise ValueError(f"{var} is marked as a parameter but it is not strong.")
        return var

    if not var.value_node.inputs:
        return None

    var_value_node = var.value_node.inputs[0]
    value_var = var_value_node.inputs[0].var
    if value_var is None:
        raise ValueError(f"{value_var=} is invalid.")
    return find_param(value_var)


def _matrix(x: Array) -> Array:
    if not np.shape(x):
        x = np.atleast_2d(x)
    elif len(np.shape(x)) == 1:
        x = np.expand_dims(x, axis=1)
    elif len(np.shape(x)) == 2:
        pass
    else:
        raise ValueError(f"Shape of x is unsupported: {np.shape(x)}")
    return x


def scaled_dot(x: Array, coef: Array, scale: Array):
    return x @ (scale * coef)


class ScaledDot(Calc):
    def __init__(
        self,
        x: Var | Node,
        coef: Var,
        scale: Var,
        _name: str = "",
    ) -> None:
        super().__init__(scaled_dot, x=x, coef=coef, scale=scale, _name=_name)
        self.update()
        self.x = x
        self.scale = scale
        self.coef = coef

    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        if not self.coef.strong:
            raise ValueError("To use predict(), coef must be a strong node.")

        coef_samples = samples[self.coef.name]
        coef_samples = np.atleast_3d(coef_samples)

        scale_samples = self.scale.predict(samples)
        scale_samples = np.atleast_3d(scale_samples)

        scaled_coef_samples = scale_samples * coef_samples

        x = x if x is not None else self.x.value
        smooth = np.tensordot(_matrix(x), scaled_coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)


class RandomIntercept(Lin):
    """
    A random intercept with iid normal prior in noncentered parameterization.
    """

    def __init__(self, x: Array, tau: Var, name: str) -> None:
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(x)
        self.x = Var.new_obs(x, name=f"{name}_covariate")
        self.basis_fn = self.label_binarizer.transform
        self.basis = Data(self.basis_fn(x), _name=f"{name}_basis")
        self.tau = tau

        prior = Dist(tfd.Normal, loc=0.0, scale=1.0)
        self.coef = Var.new_param(
            np.zeros(self.basis.value.shape[-1]), prior, name=f"{name}_coef"
        )

        self.smooth = Var(
            ScaledDot(x=self.basis, coef=self.coef, scale=self.tau),
            name=f"{name}_smooth",
        )

        tau_param = find_param(self.tau)
        self._hyper_parameters: list[str] = []
        if tau_param is not None:
            self._hyper_parameters.append(tau_param.name)

        self._parameters: list[str] = [self.coef.name]

        super(Lin, self).__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            x=self.x,
            coef=self.coef,
            tau=self.tau,
        )

        self._default_kernel = NUTSKernel
        self.mcmc_kernels: list[Kernel] = self._default_kernels()

    @property
    def hyper_parameters(self):
        return self._hyper_parameters

    @property
    def parameters(self):
        return self._parameters

    def _default_kernels(self) -> list[Kernel]:
        kernels: list[Kernel] = []

        kernels.append(self._default_kernel([self.coef.name]))

        if not self.hyper_parameters:
            return kernels

        tau_param = find_param(self.tau)
        if tau_param is not None:
            kernels.append(NUTSKernel([tau_param.name]))

        return kernels

    def ppeval(self, samples: dict, x: Array | None = None) -> Array:
        coef_samples = samples[self.coef.name]
        coef_samples = np.atleast_3d(coef_samples)

        scale_samples = self.tau.predict(samples)
        scale_samples = np.atleast_3d(scale_samples)

        scaled_coef_samples = scale_samples * coef_samples

        X = self.basis_fn(x)

        smooth = np.tensordot(X, scaled_coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)


def sumzero(nparam: int) -> Array:
    """Matrix "Z" for reparameterization for sum-to-zero-constraint."""
    j = np.ones(shape=(nparam, 1), dtype=np.float32)
    q, _ = np.linalg.qr(j, mode="complete")
    return q[:, 1:]


class RandomInterceptSumZero(RandomIntercept):
    def __init__(self, x: Array, tau: Var, name: str) -> None:
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(x)
        self.x = Var.new_obs(x, name=f"{name}_covariate")

        nparam = self.label_binarizer.transform(x).shape[-1]
        K = jnp.eye(nparam)
        Z = sumzero(nparam)
        Kz = Z.T @ K @ Z

        self.basis_fn = lambda x: self.label_binarizer.transform(x) @ Z
        self.basis = Data(self.basis_fn(x), _name=f"{name}_basis")
        self.tau = tau

        self.evals = jnp.linalg.eigvalsh(Kz)
        self.rank = Data(jnp.sum(self.evals > 0.0), _name=f"{name}_K_rank")
        _log_pdet = jnp.log(jnp.where(self.evals > 0.0, self.evals, 1.0)).sum()
        self.log_pdet = Data(_log_pdet, _name=f"{name}_K_log_pdet")

        prior = Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau,
            pen=Kz,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )

        self.coef = Var.new_param(np.zeros(Kz.shape[-1]), prior, name=f"{name}_coef")

        self.smooth = Var(
            ScaledDot(x=self.basis, coef=self.coef, scale=self.tau),
            name=f"{name}_smooth",
        )

        tau_param = find_param(self.tau)
        self._hyper_parameters: list[str] = []
        if tau_param is not None:
            self._hyper_parameters.append(tau_param.name)

        self._parameters: list[str] = [self.coef.name]

        super(Lin, self).__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            x=self.x,
            coef=self.coef,
            tau=self.tau,
        )

        self._default_kernel = NUTSKernel
        self.mcmc_kernels: list[Kernel] = self._default_kernels()
