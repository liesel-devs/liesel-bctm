from typing import Any

import jax
import jax.numpy as jnp
from liesel.model import Array, Calc, Node, Var

__docformat__ = "numpy"


__all__ = [
    "LookUpCalculator",
    "LookUp",
]


class LookUpCalculator(Calc):
    """
    A calculator that looks up :class:`float`s or ``Array``s based on a key node.

    This calculator can be used to construct nodes that take one of several discrete
    values given an indicator. The indicator is given as the ``key_node``.

    Parameters
    ----------
    mapping
        Maps float or int keys to values, which can be floats or arrays of floats. The
        array values must all have the same shape.
    key_node
        A node that provides the key for a lookup.

    Examples
    --------

    >>> import liesel.model as lsl
    >>> import numpy as np

    Look up a scalar value:

    >>> mapping = {1.0: 0.1, 2.0: 0.35}
    >>> calc = LookUpCalculator(mapping, key_node=lsl.Var(2.0))
    >>> calc.update().value
    Array(0.35, dtype=float32)

    Look up a vector:

    >>> a = np.array([1.1, 1.2])
    >>> b = np.array([2.1, 2.2])
    >>> mapping = {1.0: a, 2.0: b}
    >>> calc = LookUpCalculator(mapping, key_node=lsl.Var(2.0))
    >>> calc.update().value
    Array([2.1, 2.2], dtype=float32)

    Look up a matrix:

    >>> a = np.zeros((2, 2))
    >>> b = np.ones((2, 2))
    >>> mapping = {1.0: a, 2.0: b}
    >>> calc = LookUpCalculator(mapping, key_node=lsl.Var(2.0))
    >>> calc.update().value
    Array([[1., 1.],
           [1., 1.]], dtype=float32)

    Look up with a wrong key returns a ``NaN``:

    >>> mapping = {1.0: 0.1, 2.0: 0.35}
    >>> calc = LookUpCalculator(mapping, key_node=lsl.Var(3.0))
    >>> calc.update().value
    Array(nan, dtype=float32)

    """

    def __init__(
        self, mapping: dict[float | int, float | Array], key_node: Node | Var
    ) -> None:
        super().__init__(self._get, key_node)

        self.keys = jnp.asarray(tuple(mapping.keys()))
        self.values = jnp.asarray(tuple(mapping.values()))

        self._key_node = key_node

    def _get(self, key: float) -> Array:
        """
        Get the value corresponding to ``key``.

        Returns
        -------
        If ``key`` is a valid key, returns the value corresponding to the key.
        Otherwise, returns an array of ``NaN``s.
        """
        where = jnp.where(self.keys == key, size=1, fill_value=(-1,))[0]
        where = jnp.squeeze(where[0])

        # returns NaN if the key is not in self.keys
        value = jax.lax.cond(
            where != -1, lambda x: x, lambda x: jnp.nan * x, self.values[where]
        )

        return value


class LookUp(Node):
    """A lookup node that wraps :class:`.LookUpCalculator` for convenience."""

    def __init__(
        self,
        mapping: dict[float, float | Array],
        key_node: Node | Var,
        name: str = "",
    ) -> None:
        self.lookup_calculator = LookUpCalculator(mapping, key_node)
        super().__init__(self.lookup_calculator, _name=name)

    def update(self):
        self.lookup_calculator.update()
        return self

    @property
    def value(self) -> Any:
        return self.lookup_calculator.value
