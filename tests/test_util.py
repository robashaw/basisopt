import pytest
import pandas as pd

from basisopt.util import fit_poly, read_json
from basisopt.data import get_even_temper_params

def test_read_json():
    obj = read_json('tests/data/neon.json')
    assert type(obj).__name__ == 'AtomicBasis'
    assert 'ne' in obj.get_basis()
    shell = obj.get_basis()['ne'][0]
    assert hasattr(shell, 'exps')
    assert type(shell.exps).__name__ == 'ndarray'

def test_fit_poly():
   data = pd.read_csv('tests/data/cl2.csv')
   p, xref, re, pt = fit_poly(data['R'], data['ECC'], n=6)
   assert abs(xref - 2.00749686) < 1e-8
   assert abs(re - 1.98792829) < 1e-8
   assert len(pt) == 7
   assert abs(pt[0] + 919.45844231) < 1e-8 
   
def test_get_even_temper():
    # even_tempered_data is currently empty
    result = get_even_temper_params()
    assert len(result) == 0
    
