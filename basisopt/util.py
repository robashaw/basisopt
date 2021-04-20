# utility functions
import numpy as np

def fit_poly(x, y, n=6):
    "Fits a polynomial of order n to the set of (x [Bohr], y [Hartree]) coordinates given, and calculates data necessary for a Dunham analysis" 
    # Find best guess at minimum and shift coordinates
    xref = x[np.argmin(y)]
    xshift = x - xref
    
    # Fit polynomial to shifted system
    z = np.polyfit(xshift, y, n)
    p = np.poly1d(z)
    
    # Find the true minimum by interpolation, if possible
    xmin = min(xshift)-0.1
    xmax = max(xshift)+0.1
    crit_points = [x.real for x in p.deriv().r if np.abs(x.imag) < 1e-8 and xmin < x.real < xmax] 
    if len(crit_points) == 0:
        print("MINIMUM NOT FOUND")
        # Set outputs to default values
        re = xref
        pt = [0.0]*(n+1)
        
    else:
        dx = crit_points[0]
        re = xref + dx # Equilibrium geometry
        
        # Calculate 0th - nth Taylor series coefficients at true minimum
        pt = [p.deriv(i)(dx)/np.math.factorial(i) for i in range(n+1)]
    
    # Return fitted polynomial, x-shift, equilibrium bond length,
    # and Taylor series coefficients
    return p, xref, re, pt