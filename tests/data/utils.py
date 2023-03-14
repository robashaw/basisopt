import numpy as np


def almost_equal(x, y, thresh=1e-12):
    return np.abs(x - y) < thresh
