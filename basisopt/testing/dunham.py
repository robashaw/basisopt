# base test types
import numpy as np
from .test import Test
from basisopt.molecule import Molecule, build_diatomic
from basisopt import api, data
from basisopt.util import fit_poly
from basisopt.exceptions import InvalidDiatomic
from mendeleev import element as md_element

_VALUE_NAMES = ["E0", "R0", "BRot", "ARot", "W0", "Wx", "Wy", "De", "D0"]

def dunham(energies, distances, mu, poly_order=6, angstrom=True, Emax=0):
    "Performs a Dunham analysis on a diatomic, given energy/distance values around a minimum and the reduced mass mu"
    # convert units
    An = mu * data.FORCE_MASS
    if angstrom:
        distances *= data.TO_BOHR
    poly_order = max(poly_order, 3)
    
    # perform polynomial fit to data
    p, xref, re, pt = fit_poly(distances, energies, poly_order)
     
    # Energy at minimum, first rotational constant, and first vibrational constant
    Ee = pt[0]
    Be = 0.5 * data.TO_CM / (An * re**2)
    We = data.TO_CM * np.sqrt(2.0 * np.abs(pt[2]) / An)
    
    # Compute normalised derivatives
    npt = [(pt[i+3]/pt[2])*re**(i+1) for i in range(poly_order-2)]
    
    # Second rotational constant
    Ae = -6.0 * Be**2 * (1.0 + npt[0]) / We
    
    # First anharmonic corrections require n >= 6
    Wexe = 0.0
    Weye = 0.0
    if poly_order > 5: 
        Wexe = -1.5 * (npt[1] - 1.25*npt[0]**2) * Be
        Weye = 0.5 * (10.0*npt[3] - 35.0*npt[0]*npt[2] - 8.5*npt[1]**2 + 56.125*npt[1]*npt[0]**2 - 22.03125*npt[0]**4)*Be**2/We
    
    # Dissociation energies
    De = 0.0
    D0 = 0.0 
    if Emax != 0:
        De = (Emax - Ee) * data.TO_EV
        D0 = De - 0.5 * (We - 0.5*Wexe) * data.TO_EV/data.TO_CM
    
    return p, xref, Ee, re*data.TO_ANGSTROM, Be, Ae, We, Wexe, Weye, De, D0   

class DunhamTest(Test):
    def __init__(self, name, mol=None, mol_str="", charge=0, mult=1, poly_order=6, step=0.05, Emax=0):
        Test.__init__(self, name, mol=mol, charge=charge, mult=mult)
        self.poly_order = max(3, poly_order)
        self.step = step
        self.Emax = Emax
        
        if len(mol_str) != 0:
            self.from_string(mol_str, charge=charge, mult=mult)
            
        self.reduced_mass()
        
    def from_string(self, mol_str, charge=0, mult=1):
        self.molecule = build_diatomic(mol_str, charge=charge, mult=mult)
        
    def reduced_mass(self):
        atom1 = md_element(self.molecule._atom_names[0].title())
        atom2 = md_element(self.molecule._atom_names[1].title())
        return (atom1.mass*atom2.mass)/(atom1.mass + atom2.mass)
        
    def calculate(self, method, basis, params={}):
        if self.molecule is None:
            raise EmptyCalculation
        self.molecule.basis = basis
        self.molecule.method = method
        
        rvals = np.zeros(self.poly_order+1)
        
        midix = int(self.poly_order/2)
        rvals[midix] = self.molecule.distance(0, 1)
        for i in range(midix+1,self.poly_order+1):
            rvals[i] = rvals[i-1]+self.step
        for i in range(midix):
            rvals[midix-i-1] = rvals[midix-i] - self.step
        
        wrapper = api.get_backend()    
        energies = []
        for r in rvals:
            self.molecule._coords[0] = np.array([0.0, 0.0, -0.5*r])
            self.molecule._coords[1] = np.array([0.0, 0.0,  0.5*r])
            success = api.run_calculation(evaluate='energy', mol=self.molecule, params=params)
            energies.append(wrapper.get_value('energy'))
           
        energies = np.array(energies)
        results = dunham(energies, rvals, self.reduced_mass(), poly_order=self.poly_order, Emax=self.Emax)
        
        self.poly = results[0]
        self.shift = results[1]
        self.add_data("StencilRi", rvals*data.TO_ANGSTROM)
        self.add_data("StencilEi", energies)
        for n, r in zip(_VALUE_NAMES, results[2:]):
            self.add_data(n, r)
        
        return results[2:]
        
    
         