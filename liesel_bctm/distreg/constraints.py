import numpy as np
import scipy

from ..custom_types import Array


def ffzero(nparam1: int, nparam2: int = 1) -> Array:
    """
    Matrix "Z" for reparameterization of spline coefficients that fixes the first
    coefficient to zero.
    """
    return np.eye(nparam1 * nparam2, dtype=np.float32)[:, 1:]


def mi_sumzero(nparam1: int, nparam2: int) -> Array:
    """
    Matrix "Z" for reparameterization of spline coefficients. Using this means fixing
    the first coefficient to zero, then taking differences of adjacent coefficients
    corresponding to the first basis function evaluation of the first dimension.

    This is the constraint that is also used in the R package ``{scam}`` in the function
    ``smooth.construct.tesmi1.smooth.spec``.
    """
    Z = np.delete(np.eye(nparam1 * nparam2), obj=nparam2 - 1, axis=1)
    Z[0:nparam2, 0 : (nparam2 - 1)] = np.diff(np.identity(nparam2), 1, axis=0).T
    return Z


def z(c: Array):
    """
    Returns the matrix "Z" for reparameterization based on constraint matrix c.
    The constraint is ``c @ b = 0``, where ``b`` is the coefficient vector.
    """
    m = c.shape[0]
    q, _ = scipy.linalg.qr(c.T)
    return q[:, m:]


def sumzero(x: Array) -> Array:
    """Matrix "Z" for reparameterization for sum-to-zero-constraint."""
    j = np.ones(shape=(x.shape[0], 1), dtype=np.float32)
    C = j.T @ x
    return np.array(z(C), dtype=np.float32)
