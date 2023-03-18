# preconditioners, needs expanding
from typing import Callable

import numpy as np

Preconditioner = Callable[[np.ndarray, ...], np.ndarray]


def inverse(inv_func: Preconditioner) -> Preconditioner:
    """Decorator that adds an inverse function as an attribute
    All preconditioners must be decorated with an inverse,
    which should usually have the same signature as the parent.

    Arguments:
         inv_func (func): the inverse of the preconditioner
    """

    def decorator(func):
        func.inverse = inv_func
        return func

    return decorator


def _unit_inverse(y: np.ndarray, minval=1e-4, ratio=1.4):
    """Default, does nothing"""
    return y


@inverse(_unit_inverse)
def unit(x):
    """Identity function"""
    return x


def _positive_inverse(y, minval=1e-4, ratio=1.4):
    """Inverse of make_positive"""
    x = np.copy(y)
    for ix, v in enumerate(x):
        if v < minval:
            x[ix] = minval
            minval *= ratio
    return x


@inverse(_positive_inverse)
def make_positive(x, minval=1e-4, ratio=1.4):
    """Returns x with all values >= minval
    If multiple values are < minval, the new values
    will be minval * (ratio**n)
    """
    y = np.copy(x)
    for ix, v in enumerate(y):
        if v < minval:
            y[ix] = minval
            minval *= ratio
    return y


def _logistic_inverse(y, minval=1e-4, maxval=1e5, alpha=1.0, x0=0.0):
    """Inverse for logistic function"""
    x = (y - minval) / maxval
    x = (1.0 / x) - 1.0
    x = -np.log(x) / alpha
    return x + x0


@inverse(_logistic_inverse)
def logistic(x, minval=1e-4, maxval=1e5, alpha=1.0, x0=0.0):
    """Logistic function"""
    y = 1.0 + np.exp(-alpha * (x - x0))
    return minval + (maxval / y)
