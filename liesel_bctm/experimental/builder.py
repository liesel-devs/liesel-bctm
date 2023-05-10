from __future__ import annotations

from typing import Callable

import jax

from .. import tp_penalty as tpp
from ..builder import CTMBuilder, MITEDerivative
from ..custom_types import Array
from . import mi_splines as mi

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
    ) -> CTMBuilder:
        str_name = self._pt_name(name, "teprod_mi1_full_experimental")
        x1val, x2val = self._array(x1), self._array(x2)

        mite_spline = tpp.MIPSplineTE1(
            str_name,
            x=(x1val, x2val),
            nparam=nparam,
            a=a,
            b=b,
            order=order,
            weights=weights,
            positive_tranformation=positive_tranformation,
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
