# Wrappers for psi4 functionality
import os
import subprocess
import mendeleev as md
from basisopt.wrappers.wrapper import Wrapper, available
from basisopt.bse_wrapper import internal_basis_converter
from basisopt.exceptions import *

class OrcaWrapper(Wrapper):
    """Wrapper for Orca 5"""
    def __init__(self, orca_path):
        super(OrcaWrapper, self).__init__(name='Orca')
        self._path = orca_path
        self._method_strings = {
            'hf'            : ['energy', 'dipole', 'quadrupole'],
            'rhf'           : ['energy', 'dipole', 'quadrupole'],
            'uhf'           : ['energy', 'dipole', 'quadrupole'],
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
        molstring = f"* xyz {m.charge} {m.multiplicity}\n"
        for i in range(m.natoms()):
            molstring += m.get_line(i) + "\n"
        molstring += "*\n"
        return molstring 
    
    def _command_string(self, method, **params):
        """Helper function to turn an internal method
           name into an ORCA command line
        """
        if method == "dft":
            if "functional" in params:
                func = params["functional"]
                command = f"! {func} "
            else:
                raise KeyError("DFT functional not specified")
        else:
            command = f"! {method.upper()} "
        if "command_line" in params:
            command += params["command_line"]
        return command + "\n"
    
    def initialise(self, m, name="", tmp=".", **params):
        """Initialises Psi4 before each calculation"""
        # create input file
        self._pwd = os.getenv("PWD")
        os.chdir(tmp)
        prefix = f"{m.name}-{m.method}-" + name
        
        # handle options, molecule, basis
        cmd = self._command_string(m.method, **params)
        mol = self.convert_molecule(m)
        
        basis = "%basis\n"
        gamess_basis = internal_basis_converter(m.basis, fmt="gamess_us").split("\n")
        first_atom = True
        for line in gamess_basis[2:-1]:
            words = line.split()
            if len(words) == 1:
                if first_atom:
                    first_atom = False
                else:
                    basis += "end\n"
                atom = md.element(line.strip().title()).symbol
                basis += f"NewGTO {atom}\n"
            else:
                basis += line + "\n"
        basis += "end\nend\n"
        
        # write to file
        with open(f"{prefix}.inp", 'w') as f:
            f.write(cmd)
            f.write(mol)
            f.write(basis)
            
        return prefix
    
    def _run_orca(self, prefix, program="orca"):
        run_cmd = f"{self._path}/{program} {prefix}.inp > {prefix}.out"
        subprocess.run(run_cmd, shell=True)
        
    def _read_property_file(self, prefix, search_strings):
        with open(f"{prefix}_property.txt", 'r') as f:
            lines = f.readlines()
        
        modules = {}
        for s in search_strings:
            words = s.split(':')
            if words[0] not in modules:
                modules[words[0]] = []
            modules[words[0]].append(words[1])
        
        current_module = ""
        results = {k: None for k in search_strings}
        for line in lines:
            if "$" in line:
                current_module = line.split()[1].strip()
            elif ("#" in line) or ("---" in line):
                current_module = "" 
            elif current_module in modules:
                if ':' in line:
                    words = line.split(':')
                    name = words[0].strip()
                    key = f"{current_module}:{name}"
                    if name in modules[current_module]:
                        results[key] = words[1].strip()
                else:
                    for name in modules[current_module]:
                        if name in line:
                            key = f"{current_module}:{name}"
                            result = line.replace(name, "").strip()
                            results[key] = result
                            break
                            
        return results
    
    def _internal_clean(self, prefix):
        run_cmd = f"rm {prefix}*"
        subprocess.run(run_cmd, shell=True)
        os.chdir(self._pwd)
        
    @available
    def energy(self, mol, tmp="", **params):
        name = "energy"
        if "jobname" in params:
            name = params["jobname"]
        prefix = self.initialise(mol, tmp=tmp, name=name, **params)
        self._run_orca(prefix)
        search_string = "Calculation_Info:Total Energy"
        results = self._read_property_file(prefix, [search_string])
        self._internal_clean(prefix)
        if results[search_string] is None:
            raise FailedCalculation
        return float(results[search_string])
        
    @available
    def jk_error(self, mol, tmp="", **params):
        raise NotImplementedException
        
    @available
    def dipole(self, mol, tmp="", **params):
        raise NotImplementedException
        
    @available
    def quadrupole(self, mol, tmp="", **params):
        raise NotImplementedException
        
    @available
    def polarizability(self, mol, tmp="", **params):
        raise NotImplementedException
