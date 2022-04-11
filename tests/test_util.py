import pytest
import pandas as pd

from basisopt.util import fit_poly

def test_fit_poly():
   data = pd.read_csv('tests/data/cl2.csv')
   p, xref, re, pt = fit_poly(data['R'], data['ECC'], n=6)
   assert abs(xref - 2.00749686) < 1e-8
   assert abs(re - 1.98792829) < 1e-8
   assert len(pt) == 7
   assert abs(pt[0] + 919.45844231) < 1e-8 
    
