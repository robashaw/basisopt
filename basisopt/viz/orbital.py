# plot an orbital
import numpy as np
from mayavi import mlab
import logging

def contour3d(gto, ix=0, m=0, n=100, lower=[-2, -2, -2], upper=[2, 2, 2], contours=5):
    """Plots a 3D plot of a GTO from a Shell object
       TODO: fix creation of np.mgrid to use the (n, lower, upper) arguments
    
       Arguments:
            gto: Shell object
            ix (int): index of GTO to plot from the gto Shell
            m (int): azimuthal quantum number
            n (int): number of points per axis
            lower (list): list of lower bounds for X,Y,Z axes
            upper (list): list of upper bounds for X,Y,Z axes
            contours (int): number of contours to plot        
    """
    X, Y, Z = np.mgrid[-3:3:150j, -3:3:150j, -3:3:150j] # <- TODO
    f = gto.compute(X, Y, Z, i=ix, m=m)
    bo_logger.debug(f"Contour min: {np.min(f)}, max: {np.max(f)}")
    mlab.contour3d(X, Y, Z, f, contours=contours, colormap='cool', transparent=True)
