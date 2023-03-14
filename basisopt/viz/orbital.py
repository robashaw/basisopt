# plot an orbital
import numpy as np
from mayavi import mlab

from basisopt.containers import Shell
from basisopt.util import bo_logger


def contour3d(
    gto: Shell,
    ix: int = 0,
    m: int = 0,
    n: int = 100,
    lower: list[float] = [-2, -2, -2],
    upper: list[float] = [2, 2, 2],
    contours: int = 5,
) -> object:
    """Plots a 3D plot of a GTO from a Shell object

    Arguments:
         gto: Shell object
         ix (int): index of GTO to plot from the gto Shell
         m (int): azimuthal quantum number
         n (int): number of points per axis
         lower (list): list of lower bounds for X,Y,Z axes
         upper (list): list of upper bounds for X,Y,Z axes
         contours (int): number of contours to plot

    Returns:
         the mayavi figure object
    """
    nj = n * 1j
    X, Y, Z = np.mgrid[
        (lower[0]) : (upper[0]) : nj, (lower[1]) : (upper[1]) : nj, (lower[2]) : (upper[2]) : nj
    ]
    f = gto.compute(X, Y, Z, i=ix, m=m)
    bo_logger.debug("Contour min: %12.6f, max: %12.6f", np.min(f), np.max(f))
    return mlab.contour3d(X, Y, Z, f, contours=contours, colormap='cool', transparent=True)
