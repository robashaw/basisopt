# Wrappers for psi4 functionality
from basisopt.wrappers.wrapper import Wrapper, available
from basisopt.bse_wrapper import internal_basis_converter
from basisopt.exceptions import *
import psi4

class Psi4Wrapper(Wrapper):
    def __init__(self):
        Wrapper.__init__(self, name='Psi4')
        self._method_strings = {
            'scf'       : ['energy', 'dipole', 'quadrupole'],
            'hf'        : ['energy', 'dipole', 'quadrupole'],
            'dft'       : ['energy', 'dipole', 'quadrupole'],
            'mp2'       : ['energy', 'dipole', 'quadrupole'],
            'mp3'       : ['energy'],
            'mp4'       : ['energy'],
            'lccd'      : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'lccsd'     : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cc2'       : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cc3'       : ['energy'],
            'ccsd'      : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'ccsd(t)'   : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cisd'      : ['energy', 'trans_dipole', 'trans_quadrupole'],
            'ras-ci'    : ['energy'],
            'casscf'    : ['energy', 'trans_dipole', 'trans_quadrupole'],
            'rasscf'    : ['energy', 'trans_dipole', 'trans_quadrupole'],
            'adc(1)'    : ['energy', 'dipole', 'trans_dipole'],
            'adc(2)'    : ['energy', 'dipole', 'trans_dipole'],
            'adc(3)'    : ['energy', 'dipole', 'trans_dipole'],
        }
        
    def convert_molecule(self, m):
        molstring = ""
        for i in range(m.natoms()):
            molstring += m.get_line(i) + "\n"
        return psi4.geometry(molstring)
    
    def _property_prefix(self, method):
        m = method.lower()
        if m in ['scf', 'hf', 'dft']:
            return 'SCF'
        elif m in ['cisd']:
            return 'CI'
        else:
            return method.upper()
    
    def initialise(self, m, name="", tmp=""):
        outfile = tmp + f"{m.name}-{m.method}-" + name + ".out"
        psi4.core.set_output_file(outfile, False)
        
        mol = self.convert_molecule(m)
        mol.set_molecular_charge(m.charge)
        mol.set_multiplicity(m.multiplicity)
        
        # logic to check global options
        if "memory" in self._globals:
            psi4.set_memory(self._globals["memory"])
        psi4.set_options({k: v for k, v in self._globals.items() if k != "memory"})
        
        # set basis
        g94_basis = internal_basis_converter(m.basis, fmt="psi4")
        psi4.basis_helper(g94_basis) 
    
    def _get_properties(self, mol, name="prop", properties=[], tmp=""):
        ptype = self._property_prefix(mol.method)
        strings = [ptype + " " + p.upper() for p in properties]
        
        if len(strings) == 0:
            raise EmptyCalculation
                
        self.initialise(mol, name=name, tmp=tmp)
        E, wfn = psi4.properties(mol.method, return_wfn=True, properties=properties)
        
        results = {}
        for p, s in zip(properties, strings):
            if s in wfn.variables():
                results[p] = wfn.variable(s)
            else:
                raise PropertyNotAvailable(p)
        
        return results
        
    @available
    def energy(self, mol=None, tmp=""):
        self.initialise(mol, name="energy", tmp=tmp)
        runstring = f"{mol.method}"
        return psi4.energy(runstring)
        
    @available
    def dipole(self, mol=None, tmp=""):
        results = self._get_properties(mol, name="dipole", properties=['dipole'], tmp=tmp)
        return results['dipole']
        
    @available
    def quadrupole(self, mol=None, tmp=""):
        results = self._get_properties(mol, name="quadrupole", properties=['quadrupole'], tmp=tmp)
        return results['quadrupole']
        
    @available
    def polarizability(self, mol=None, tmp=""):
        results = self._get_properties(mol, name="polar", properties=['polarizability'], tmp=tmp)
        return results['polarizability']
