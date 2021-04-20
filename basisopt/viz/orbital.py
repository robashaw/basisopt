# plot an orbital
import numpy as np
from mayavi import mlab

def contour3d(gto, n=100, lower=[-2, -2, -2], upper=[2, 2, 2], contours=5):
    X, Y, Z = np.mgrid[-3:3:150j, -3:3:150j, -3:3:150j]
    f = gto.compute(X, Y, Z)
    print(np.min(f), np.max(f))
    mlab.contour3d(X, Y, Z, f, contours=contours, colormap='cool', transparent=True)