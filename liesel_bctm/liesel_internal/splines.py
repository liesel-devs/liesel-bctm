from functools import partial

import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap
from liesel.distributions import MultivariateNormalDegenerate
from liesel.model import Array, Calc, Dist, Group, Var
from tensorflow_probability.substrates.jax.distributions import InverseGamma


def _check_equidistant_knots(knots: Array) -> Array:
    """
    Checks if knots are equidistants.
    """
    diff = jnp.diff(knots)

    return jnp.allclose(diff, diff[0], 1e-3, 1e-3)


def _check_data_range(x: Array, knots: Array, order: int) -> bool:
    """
    Check that values in x are in the range
    [knots[order], knots[dim(knots) - order - 1]].
    """

    return (
        jnp.min(x) >= knots[order] and jnp.max(x) <= knots[knots.shape[0] - order - 1]
    )


def _check_b_spline_inputs(x: Array, knots: Array, order: int) -> None:
    if not order >= 0:
        raise ValueError("Order must non-negative")
    if not _check_equidistant_knots(knots):
        raise ValueError("Sorted knots are not equidistant")
    # if not _check_data_range(x, knots, order):
    #     raise ValueError(
    #         f"Data values are not in the range \
    #             [{knots[order]}, {knots[knots.shape[0] - order - 1]}]"
    #     )


# @jit
@partial(jit, static_argnums=(1, 2))
def create_equidistant_knots(x: Array, order: int = 3, n_params: int = 20) -> Array:
    """
    Create equidistant knots for B-Spline of the specified order.

    Some additional info:

    - ``dim(knots) = n_params + order + 1``
    - ``n_params = dim(knots) - order - 1``

    Parameters
    ----------
    x
        The data for which the knots are created.
    order
        A positive integer giving the order of the spline function.
        A cubic spline has order of 3.
    n_params
        Number of parameters of the B-spline.

    """
    epsilon = 0.01

    internal_k = n_params - order + 1

    a = jnp.min(x)
    b = jnp.max(x)

    min_k = a - jnp.abs(a * epsilon)
    max_k = b + jnp.abs(b * epsilon)

    internal_knots = jnp.linspace(min_k, max_k, internal_k)

    step = internal_knots[1] - internal_knots[0]

    left_knots = jnp.linspace(min_k - (step * order), min_k - step, order)
    right_knots = jnp.linspace(max_k + step, max_k + (step * order), order)

    return jnp.concatenate((left_knots, internal_knots, right_knots))


@partial(jit, static_argnums=2)
def _build_basis_vector(x: Array, knots: Array, order: int) -> Array:
    """
    Builds a vector of length (dim(knots) - order - 1).
    Each entry i is iterativaly updated. At time m,
    the entry i is the evaluation of the basis function
    at the observed value for the m-th order and for the i-th knot.
    The creation of the matrix needs a row-wise (order) loop (f1)
    and a column-wise (knot index) loop (f2).
    """
    k = knots.shape[0] - order - 1
    bv = jnp.full(knots.shape[0] - 1, jnp.nan)

    def basis_per_order(m, bv):
        def basis_per_knot(i, bv):
            def base_case(bv):
                return bv.at[i].set(
                    jnp.where(x >= knots[i], 1.0, 0.0)
                    * jnp.where(x < knots[i + 1], 1.0, 0.0)
                )

            def recursive_case(bv):
                b1 = (x - knots[i]) / (knots[i + m] - knots[i]) * bv[i]
                b2 = (
                    (knots[i + m + 1] - x)
                    / (knots[i + m + 1] - knots[i + 1])
                    * bv[i + 1]
                )

                return bv.at[i].set(b1 + b2)

            return lax.cond(m == 0, base_case, recursive_case, bv)

        return lax.fori_loop(0, k + order, basis_per_knot, bv)

    return lax.fori_loop(0, order + 1, basis_per_order, bv)[:k]


def build_design_matrix_b_spline(x: Array, knots: Array, order: int) -> Array:
    """
    Builds the design matrix for B-Splines of the specified order
    defined by the knots at the values in x. Instead of applying the recursive
    definition of B-splines, a matrix of (order + 1) rows and (dim(knots) - order - 1)
    columns for each value in x is created. This matrix store the evaluation of
    the basis function at the observed value for the m-th order and for the i-th knot.
    """
    knots = jnp.sort(knots)

    _check_b_spline_inputs(x, knots, order)

    return _build_design_matrix_b_spline_aux(x, knots, order)


@partial(jit, static_argnums=2)
def _build_design_matrix_b_spline_aux(x: Array, knots: Array, order: int) -> Array:
    """
    Fills the design matrix taking the values in the order-th row and of the first
    (dim(knots) - order - 1) columns from the output of the build_basis_matrix function
    called for each data point.
    """
    return vmap(lambda x: _build_basis_vector(x, knots, order))(x)


def _check_p_spline_matrix_inputs(d: int, r: int) -> None:
    if not d > 0:
        raise ValueError("Matix dimension must be positive")
    if not r >= 0:
        raise ValueError("Difference order must be non-negative")


def build_p_spline_matrix(d: int, r: int = 1) -> Array:
    _check_p_spline_matrix_inputs(d, r)

    return _build_p_spline_matrix_aux(d, r)


@partial(jit, static_argnums=(0, 1))
def _build_p_spline_matrix_aux(d: int, r: int = 1) -> Array:
    """
    Builds (d x d) a penalty matrix with differences of order r.
    """
    D = jnp.diff(jnp.identity(d), r, axis=0)

    return D.T @ D


class PenaltyMatrix(Var):
    """A strong node representing the penalty matrix of a P-Spline."""


def model_p_spline(
    X: Array,
    K: Array,
    a: float,
    b: float,
    beta: Array,
    name: str | None = None,
) -> Group:
    """Returns a InputGroup representing a B-Spline."""

    X_obs = Var.new_obs(X, name="X")
    K_var = Var(K, name="K")
    tau_a = Var(a, name="a")
    tau_b = Var(b, name="b")

    rank = Var(jnp.linalg.matrix_rank(K), name="rank")
    tau2_dist = Dist(InverseGamma, concentration=tau_a, scale=tau_b)
    tau2 = Var(10000.0, tau2_dist, name="tau2")

    prior = Dist(
        MultivariateNormalDegenerate.from_penalty,
        loc=0.0,
        var=tau2,
        pen=K_var,
        rank=rank,
    )

    beta_param = Var.new_param(value=beta, distribution=prior, name="beta")

    f = Var(Calc(lambda X, beta: X @ beta, X=X_obs, beta=beta_param), name="f")

    return Group(name=str(name), X=X_obs, beta=beta_param, K=K_var, tau2=tau2, f=f)


_build_basis_vector_derivative = jacfwd(_build_basis_vector)


def build_design_matrix_b_spline_derivative(
    x: Array, knots: Array, order: int
) -> Array:
    """
    Builds a design matrix of B-Spline derivative evaluations.

    Parameters
    ===========
    x
        Input data, should be a 1d array.
    knots
        An array of knots, including inner and outer knots.
    order
        Spline order.

    Returns
    =======
    Array
        A 2d array representing a design matrix of B-spline derivative evaluations.
        The array has dimensions ``(n, d)``, where ``n = len(x)`` and
        ``d = len(knots) + order + 1``.
    """
    knots = jnp.sort(knots)

    _check_b_spline_inputs(x, knots, order)

    return _build_design_matrix_b_spline_derivative_aux(x, knots, order)


@partial(jit, static_argnums=2)
def _build_design_matrix_b_spline_derivative_aux(
    x: Array, knots: Array, order: int
) -> Array:
    """
    Fills the design matrix taking the values in the order-th row and of the first
    (dim(knots) - order - 1) columns from the output of the
    build_basis_matrix_derivative function called for each data point.

    Docstring copied and adapted from :func:`._build_design_matrix_b_spline_aux`.
    """
    return vmap(lambda x: _build_basis_vector_derivative(x, knots, order))(x)
