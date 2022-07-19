import pytest
import numpy as np

from basisopt.basis.basis import *
from tests.data.shells import get_vdz_internal
from tests.data.utils import almost_equal

def test_uncontract_shell():
    shell = Shell()
    shell.exps = np.array([0.1, 1.0, 2.0, 4.0, 8.0])
    shell.coefs = [
        np.array([0.5, -0.5, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, -0.5, -0.5])
    ]
    uncontract_shell(shell)
    assert len(shell.coefs) == len(shell.exps)
    for c in shell.coefs:
        assert np.sum(c) == 1.
    
def test_uncontract():
    vdz = get_vdz_internal()
    
    new_vdz = uncontract(vdz, elements=['o'])
    assert 'o' not in new_vdz
    
    new_vdz = uncontract(vdz)
    for s in new_vdz['h']:
        assert len(s.exps) == len(s.coefs)
    
def test_even_temper_expansion():
    et_params = [(1.5, 1.9, 15), (2.7, 1.6, 12)]
    et_basis  = even_temper_expansion(et_params)
    assert len(et_basis) == 2
    
    s_shell = et_basis[0]
    p_shell = et_basis[1]
    assert len(s_shell.exps) == 15
    assert almost_equal(s_shell.exps[0], 1.5, thresh=1e-6)
    assert almost_equal(s_shell.exps[6], 70.568822, thresh=1e-6)
    assert len(p_shell.exps) == 12
    assert almost_equal(p_shell.exps[11], 474.989023, thresh=1e-6)
    
def test_fix_ratio():
    exps = np.array([0.1, 0.2, 0.58, 1.3, 3.0, 8.2])
    new_exps = fix_ratio(exps)
    assert almost_equal(np.sum(exps - new_exps), 0.)
    
    new_exps = fix_ratio(exps, 2.4)
    expected = np.array([0.1, 0.24, 0.58, 1.392, 3.3408, 8.2])
    assert almost_equal(np.sum(expected - new_exps), 0.)
    
def test_basis_init():
    pass
    
def test_basis_load():
    pass
    
def test_basis_tests():
    pass
