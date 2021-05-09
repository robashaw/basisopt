from basisopt import Molecule
from basisopt.wrappers import Wrapper
from basisopt.wrappers.dummy import DummyWrapper
from basisopt.exceptions import InvalidMethodString
import pytest
from data.utils import almost_equal
from data.shells import get_vdz_internal
import numpy as np

def test_empty_wrappers():
    w = Wrapper()
    assert w._name == 'Empty'
    assert len(w._methods.keys()) > 0
    
    dw = DummyWrapper()
    assert dw._name == 'Dummy'
    assert len(dw._methods.keys()) > 0
    
def test_add_global():
    w = Wrapper()
    w.add_global("Memory", "1gb")
    assert "Memory" in w._globals
    assert w._globals["Memory"] == "1gb"
    
def test_get_value():
    w = Wrapper()
    assert w.get_value("energy") is None
    w._values['energy'] = 1
    assert w.get_value("energy") == 1

def test_verify_method_string():
    w = Wrapper()
    assert not w.verify_method_string("rhf.energy")
    
    dw = DummyWrapper()
    assert dw.verify_method_string("linear.energy")
    assert dw.verify_method_string("uniform.dipole")
    assert not dw.verify_method_string("quadratic.quadrupole")
    assert not dw.verify_method_string("exp.polarizability")
    
    with pytest.raises(InvalidMethodString):
        w.verify_method_string("rhfenergy")
        dw.verify_method_string("uniform/dipole")

def test_run():
    m = Molecule()
    m.from_xyz("data/caffeine.xyz") # 24 atoms
    
    w = Wrapper()
    assert w.run("energy", m, {}) == -1
    
    dw = DummyWrapper()
    m.method = "linear"
    assert dw.run("energy", m, {}) == 0
    assert almost_equal(dw.get_value("energy"), -24) 
     
    m.method = "exp"
    assert dw.run("dipole", m, {}) == 0
    assert almost_equal(dw.get_value("dipole"), 81377.395709502, thresh=1e-8)
    assert dw.run("quadrupole", m, {}) == -1
    
def test_method_is_available():
    w = Wrapper()
    assert not w.method_is_available()
    assert not w.method_is_available(method="dipole")
    assert not w.method_is_available(method="trans_dipole")
    assert not w.method_is_available(method="cheese")
    
    dw = DummyWrapper()
    assert dw.method_is_available()
    assert dw.method_is_available(method="dipole")
    assert not dw.method_is_available(method="trans_dipole")
    assert not dw.method_is_available(method="cheese")
    
def test_all_available():
    w = Wrapper()
    assert len(w.all_available) == 0
    assert len(w.available_properties("linear")) == 0
    assert len(w.available_methods("energy")) == 0
    
    dw = DummyWrapper()
    assert len(dw.all_available) == 4
    assert len(dw.available_properties("linear")) == 4
    assert len(dw.available_methods("energy")) == 4
    
def test_initialise():
    dw = DummyWrapper()
    m = Molecule()
    m.from_xyz("data/caffeine.xyz")
    
    dw.initialise(m)
    assert dw._value == 24
    assert not dw._memory_set
    assert dw._basis_value == 0
    
    dw.add_global("memory", "1gb")
    m.basis = get_vdz_internal()
    assert not dw._memory_set
    dw.initialise(m)
    assert dw._value == 24
    assert dw._memory_set
    assert dw._basis_value == 1