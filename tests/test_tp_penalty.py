import numpy as np

from liesel_bctm.custom_types import Array
from liesel_bctm.distreg import psplines as ps
from liesel_bctm.tp_penalty import cov_pen


def test_cov_pen() -> Array:
    pen = cov_pen(5, 5)
    assert pen.shape == (25, 25)

    assert np.allclose(pen[0:5, 0:5], ps.pen(5))  # 2nd differences part
    assert np.allclose(pen[5:10, 5:10], ps.pen(5, diff=1))  # 1st differences part
