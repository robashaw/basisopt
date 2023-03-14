# regularisers, needs expanding
from typing import Callable

import numpy as np

Regulariser = Callable[[np.ndarray], float]


def l1_norm(x: np.ndarray):
    return np.sum(np.abs(x))


def l2_norm(x: np.ndarray):
    return np.linalg.norm(x)


def linf_norm(x: np.ndarray):
    return np.amax(np.abs(x))
