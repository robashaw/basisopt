# Wrappers for psi4 functionality
import cclib
import os
import subprocess
from basisopt.wrappers.wrapper import Wrapper, available
from basisopt.bse_wrapper import internal_basis_converter
from basisopt.exceptions import *

class OrcaWrapper(Wrapper):
    """Wrapper for Orca 5"""
    def __init__(self, orca_path):
        Wrapper.__init__(self, name='Orca')
        self._path = orca_path
        self._method_strings = {
            'scf'           : ['energy', 'dipole', 'quadrupole'],
            'hf'            : ['energy', 'dipole', 'quadrupole'],
            'dft'           : ['energy', 'dipole', 'quadrupole'],
            'mp2'           : ['energy', 'dipole', 'quadrupole'],
            'scs-mp2'       : ['energy', 'dipole', 'quadrupole'],
            'dlpno-mp2'     : ['energy', 'dipole', 'quadrupole'],
            'ccsd'          : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'dlpno-ccsd'    : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'ccsd(t)'       : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'dlpno-ccsd(t)' : ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cisd'          : ['energy', 'trans_dipole', 'trans_quadrupole'],
            'casscf'        : ['energy', 'trans_dipole', 'trans_quadrupole'],
            'rasscf'        : ['energy', 'trans_dipole', 'trans_quadrupole']
        }
        
    def convert_molecule(self, m):
        """Convert an internal Molecule object
           to an Orca geometry section
        """
        molstring = f"* int {m.charge} {m.multiplicity}"
        for i in range(m.natoms()):
            molstring += m.get_line(i) + "\n"
        molstring += "*\n"
        return molstring 
    
    def _command_string(self, method, **params):
        """Helper function to turn an internal method
           name into an ORCA command line
        """
        
    
    def initialise(self, m, name="", tmp="", **params):
        """Initialises Psi4 before each calculation"""
        # create input file
        self._pwd = os.getenv("PWD")
        os.chdir(tmp)
        prefix = f"{m.name}-{m.method}-" + name
        
        # handle options, molecule, basis
        cmd = self._command_string(m.method, **params)
        mol = self.convert_molecule(m
        gamess_basis = internal_basis_converter(m.basis, fmt="gamess_us")
        
        # write to file
        with open(f"{prefix}.inp", 'w') as f:
            f.write(cmd)
            f.write(mol)
            f.write(gamess_basis)
            f.close()
    
    def _run_orca(self, prefix):
        run_cmd = f"{self._path} {prefix}.inp > {prefix}.out"
        subprocess.run(run_cmd)
    
    def clean(self, prefix):
        run_cmd = f"rm {prefix}*"
        subprocess.run(run_cmd)
        os.chdir(self._pwd)
        
    @available
    def energy(self, mol, tmp=""):

        
    @available
    def dipole(self, mol, tmp=""):

        
    @available
    def quadrupole(self, mol, tmp=""):

        
    @available
    def polarizability(self, mol, tmp=""):

