from basisopt.molecule import *
from basisopt.exceptions import InvalidDiatomic
import pytest
from tests.data.utils import almost_equal
import numpy as np

def test_default_molecule():
    m = Molecule()
    assert m.name == "Untitled"
    assert m.charge == 0
    assert m.multiplicity == 1
    assert m.method == ""
    assert len(m.basis.keys()) == 0
    assert len(m.unique_atoms()) == 0
    assert m.natoms() == 0
    
def test_nelectrons():
    m = Molecule()
    nel = m.nelectrons()
    assert m.nelectrons() == 0
    m = Molecule.from_xyz("tests/data/caffeine.xyz")
    assert m.nelectrons() == 102
    
def test_add_atom():
    m = Molecule()
    
    m.add_atom()
    assert m.natoms() == 1
    assert 'H' in m._atom_names
    
    m.add_atom(element='O', coord=[1.5, 0.0, 0.0])
    assert m.natoms() == 2
    assert 'O' in m._atom_names
    assert almost_equal(m.distance(0, 1), 1.5)
    
def test_add_get_result():
    m = Molecule()
    m.add_result("energy", -0.5)
    assert almost_equal(m.get_result("energy"), -0.5)
    m.add_result("energy", -0.48)
    assert m.get_result("energy") > -0.49
    assert almost_equal(m.get_result("force"), 0.0)
    
def test_get_delta():
    m = Molecule()
    m.add_result("energy", -0.48)
    m.add_reference("energy", -0.5)
    assert almost_equal(m.get_delta("energy"), 0.02)
    
    m.add_reference("cheese", 1)
    assert almost_equal(m.get_delta("cheese"), -1)
    
def test_from_xyz():
    m = Molecule.from_xyz("tests/data/caffeine.xyz",
                          name="Caffeine", charge=1, mult=3)
    assert m.natoms() == 24
    assert m.name == "Caffeine"
    assert m.charge == 1
    assert m.multiplicity == 3
    
    # test unique_atoms
    unique_atoms = m.unique_atoms()
    assert 'H' in unique_atoms
    assert 'C' in unique_atoms
    assert 'N' in unique_atoms
    assert 'O' in unique_atoms
    assert 'S' not in unique_atoms
    
    # test distance
    assert almost_equal(m.distance(0, 5), 3.3795237480990497)
    assert almost_equal(m.distance(1, 3), 2.4636465289488934)
    assert almost_equal(m.distance(6, 22), 4.250292507670527)
    assert almost_equal(m.distance(2, 2), 0.0)

    # test get_line
    line1 = "H\t-3.380413\t-1.1272367\t0.5733036"
    line2 = "N\t0.9668296\t-1.0737425\t-0.8198227"
    line24 = "H\t-1.2074498\t2.7537592\t1.7203047"
    assert m.get_line(1) == line2
    assert m.get_line(23) == line24
    assert m.get_line(44) == line24
    assert m.get_line(-3) == line1
    
def test_build_diatomic():    
    no = build_diatomic("NO,1.3", charge=1)
    assert no.natoms() == 2
    assert almost_equal(no.distance(0, 1), 1.3)
    assert no.charge == 1
    assert no.multiplicity == 1
    assert len(no.unique_atoms()) == 2
    
    h2 = build_diatomic("H2,0.9")
    assert h2.natoms() == 2
    assert h2.charge == 0
    assert len(h2.unique_atoms()) == 1
    
    lih = build_diatomic("LiH,1.1", mult=3)
    assert lih.natoms() == 2
    assert lih.multiplicity == 3
    
    ne2 = build_diatomic("Ne2,3.0")
    assert ne2.natoms() == 2
    assert len(ne2.unique_atoms()) == 1
    assert almost_equal(ne2.distance(0, 1), 3.0)
    
    licl = build_diatomic("LiCl,1.8") 
    assert licl.natoms()==2
    assert len(licl.unique_atoms())==2
    
    with pytest.raises(IndexError):
        m = build_diatomic("H2")
        
    with pytest.raises(InvalidDiatomic):
        m1 = build_diatomic("H2O,1.4")
        m2 = build_diatomic("Ne,1.4")
        m3 = build_diatomic("C5,1.4")
        m4 = build_diatomic("CHCl3,1.4")
    
        
        
    
