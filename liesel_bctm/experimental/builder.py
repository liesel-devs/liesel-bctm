from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from ..liesel_internal import splines

from .. import tp_penalty as tpp
from ..builder import CTMBuilder, MITEDerivative
from ..custom_types import Array
from ..distreg.mi_splines import MIPSplineTE1
from . import mi_splines as mi

kn = splines.create_equidistant_knots

# flake8: noqa


class ExperimentalCTMBuilder(CTMBuilder):
    def add_teprod_mi1_full(
        self,
        x1: Array | str,
        x2: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        weights: Array | None = None,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
        knot_boundaries: tuple[
            tuple[float, float] | None, tuple[float, float] | None
        ] = (None, None),
    ) -> CTMBuilder:
        str_name = self._pt_name(name, "teprod_mi1_full_experimental")
        x1val, x2val = self._array(x1), self._array(x2)

        bounds_y, bounds_x = knot_boundaries
        if bounds_y is not None:
            knots_y = kn(jnp.array(bounds_y), order=order, n_params=nparam[0])
        else:
            knots_y = None

        if bounds_x is not None:
            knots_x = kn(jnp.array(bounds_x), order=order, n_params=nparam[1])
        else:
            knots_x = None

        mite_spline = MIPSplineTE1(
            str_name,
            x=(x1val, x2val),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            weights=weights,
            positive_tranformation=positive_tranformation,
            knots=(knots_y, knots_x),
            Z=None,
        )
        self.pt.append(mite_spline)
        self.add_groups(mite_spline)
        return self

    def add_teprod_exp2(
        self,
        y: Array | str,
        x: Array | str,
        nparam: tuple[int, int],
        a: float,
        b: float,
        weights: Array | None = None,
        order: int = 3,
        positive_tranformation: Callable[[Array], Array] = jax.nn.softplus,
        name: str | None = None,
    ) -> ExperimentalCTMBuilder:
        str_name = self._pt_name(name, "teprod_mi1_full_experimental")
        x1val, x2val = self._array(y), self._array(x)

        mite_spline = mi.ExperimentalTE2(
            str_name,
            x=(x1val, x2val),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            weights=weights,
            positive_tranformation=positive_tranformation,
        )
        self.pt.append(mite_spline)
        self.add_groups(mite_spline)
        mipsd = MITEDerivative(mite_spline, name=str_name + "_d")
        self.ptd.append(mipsd)

        return self
