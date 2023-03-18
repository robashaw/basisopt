# Wrappers for psi4 functionality
import os
import subprocess
from typing import Any

import mendeleev as md
import numpy as np

from basisopt.bse_wrapper import internal_basis_converter
from basisopt.containers import InternalBasis
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.wrappers.wrapper import Wrapper, available


class OrcaWrapper(Wrapper):
    """Wrapper for Orca 5

    Private attribute:
         pwd (str): the present working directory
    """

    def __init__(self, orca_path: str):
        """Args:
        orca_path (str): path to the orca executable
        """
        super().__init__(name='Orca')
        self._path = orca_path
        self._method_strings = {
            'hf': ['energy', 'dipole', 'quadrupole', 'polarizability', 'jk_error'],
            'rhf': ['energy', 'dipole', 'quadrupole', 'polarizability', 'jk_error'],
            'uhf': ['energy', 'dipole', 'quadrupole', 'polarizability', 'jk_error'],
            'dft': ['energy', 'dipole', 'quadrupole', 'polarizability', 'jk_error'],
            'mp2': ['energy', 'dipole', 'quadrupole'],
            'scs-mp2': ['energy', 'dipole', 'quadrupole'],
            'dlpno-mp2': ['energy', 'dipole', 'quadrupole'],
            'ccsd': ['energy', 'dipole', 'quadrupole'],
            'dlpno-ccsd': ['energy', 'dipole', 'quadrupole'],
            'ccsd(t)': ['energy', 'dipole', 'quadrupole'],
            'dlpno-ccsd(t)': ['energy', 'dipole', 'quadrupole'],
            'cisd': ['energy', 'trans_dipole', 'trans_quadrupole'],
            'casscf': ['energy', 'trans_dipole', 'trans_quadrupole'],
            'rasscf': ['energy', 'trans_dipole', 'trans_quadrupole'],
        }
        self._pwd = "."

    def convert_molecule(self, m: Molecule) -> str:
        """Convert an internal Molecule object
        to an Orca geometry section
        """
        molstring = f"* xyz {m.charge} {m.multiplicity}\n"
        for i in range(m.natoms()):
            molstring += m.get_line(i) + "\n"
        molstring += "*\n"
        return molstring

    def _density_prefix(self, method: str) -> str:
        """Grabs the prefix for density options
        for a given method.
        """
        if "mp2" in method:
            return "mp2"
        if ("cc" in method) or ("ci" in method):
            return "mdci"
        return None

    def _command_string(self, method: str, **params) -> str:
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

        if "density" in params:
            prefix = self._density_prefix(method)
            if prefix:
                command += f"\n%{prefix}\n"
                command += "density\t" + params["density"]
                command += "\nend"

        if "elprop" in params:
            command += "\n%elprop\n"
            for line in params["elprop"]:
                command += line + "\n"
            command += "end"

        return command + "\n"

    def _convert_basis(self, basis: InternalBasis, gto_string: str = "NewGTO") -> str:
        """Converts an InternalBasis to the basis string for ORCA 5

        Arguments:
            basis (InternalBasis): basis set to convert
            gto_string (str): NewGTO for orbital, NewAuxGTO for just coulomb,
                NewAuxJKGTO for coulomb+exchange

        Returns:
            orca basis block string
        """
        gamess_basis = internal_basis_converter(basis, fmt="gamess_us").split("\n")
        first_atom = True
        basis_str = ""
        for line in gamess_basis[2:-1]:
            words = line.split()
            if len(words) == 1:
                if first_atom:
                    first_atom = False
                else:
                    basis_str += "end\n"
                atom = md.element(line.strip().title()).symbol
                basis_str += f"{gto_string} {atom}\n"
            else:
                basis_str += line + "\n"
        basis_str += "end\n"
        return basis_str

    def initialise(self, m: Molecule, name: str = "", tmp: str = ".", **params) -> str:
        """Initialises Orca and creates input file for calculation.
        WARNING: moves to the scratch directory, but saves PWD in
                self._pwd

        Arguments:
                m (Molecule): molecule to run calculation on
                name (str): name for calculation
                tmp (str): path to scratch directory

        Returns:
                the prefix for the calculation input/output files
        """
        # create input file
        self._pwd = os.getenv("PWD")
        os.chdir(tmp)
        prefix = f"{m.name}-{m.method}-" + name

        # handle options, molecule, basis
        cmd = self._command_string(m.method, **params)
        mol = self.convert_molecule(m)

        basis = "%basis\n"
        basis += self._convert_basis(m.basis)
        if m.jkbasis:
            basis += self._convert_basis(m.jkbasis, gto_string="NewAuxJKGTO")
        elif m.jbasis:
            basis += self._convert_basis(m.jbasis, gto_string="NewAuxJGTO")
        basis += "end\n"

        # write to file
        with open(f"{prefix}.inp", 'w', encoding='utf-8') as f:
            f.write(cmd)
            f.write(mol)
            f.write(basis)

        return prefix

    def _run_orca(self, prefix: str, program: str = "orca"):
        """Run an ORCA executable

        Arguments:
               prefix (str): prefix for input/output files
               program (str): the orca executable to use
        """
        run_cmd = f"{self._path}/{program} {prefix}.inp > {prefix}.out"
        subprocess.run(run_cmd, shell=True, check=True)

    def _read_property_file(self, prefix: str, search_strings: list[str]) -> dict[str, Any]:
        """Reads in desired results from the orca [name]_property.txt file.

        Arguments:
            prefix (str): the prefix for the input/ouput files
            search_strings (list[str]): a list of search strings of the form
                    "{block}:{property}", e.g. "Calculation_Info:Total Energy"
                    would return the Total Energy value from the Calculation_Info
                    block of the property file

        Returns:
            a dictionary of {search_string}:{value}
        """
        # read in the property file
        with open(f"{prefix}_property.txt", 'r') as f:
            lines = f.readlines()

        # break the search strings up by module
        modules = {}
        special_keys = []
        for s in search_strings:
            words = s.split(':')
            if words[0] not in modules:
                if "Electric_Properties" in words[0]:
                    # Dipoles/Quadrupoles are vectors/matrices
                    special_keys.append(words[0])
                modules[words[0]] = []
            modules[words[0]].append(words[1])

        # Go through the property file tracking down the
        # modules and properties in modules dict
        current_module = ""
        results = {k: None for k in search_strings}
        for ix, line in enumerate(lines):
            if "$" in line:
                current_module = line.split()[1].strip()
            elif ("#" in line) or ("---" in line):
                current_module = ""
            elif current_module in modules:
                # annoyingly, some properties have colons after the property name
                # while others just have a space, because ORCA is horribly
                # inconsistent about how it prints things out
                if ':' in line:
                    words = line.split(':')
                    name = words[0].strip()
                    key = f"{current_module}:{name}"
                    if name in modules[current_module]:
                        if "Electric_Properties" in current_module:
                            # place the line index so that subsequent
                            # lines can be parsed
                            results[key] = ix
                        else:
                            results[key] = words[1].strip()
                else:
                    for name in modules[current_module]:
                        if name in line:
                            key = f"{current_module}:{name}"
                            if "Electric_Properties" in current_module:
                                results[key] = ix
                            else:
                                result = line.replace(name, "").strip()
                                results[key] = result
                            break

        # to grab electric properties, we need to reloop over
        # certain lines
        for key in special_keys:
            module = modules[key]
            for name in module:
                line_ix = results[f"{key}:{name}"]
                if "Dipole" in name:
                    # 3D vector
                    res = np.zeros(3)
                    for i, line in enumerate(lines[line_ix + 2 : line_ix + 5]):
                        res[i] = float(line.split()[1])
                elif "quadrupole" in name:
                    # 3x3 matrix
                    res = np.zeros([3, 3])
                    for i, line in enumerate(lines[line_ix + 2 : line_ix + 5]):
                        row = [float(w) for w in line.split()[1:]]
                        res[i] = np.array(row)
                elif "polarizability" in name:
                    # currently taking the isotropic polarizability
                    # could change to take raw tensor
                    line = lines[line_ix]
                    res = line.split(':')[1].strip()
                results[f"{key}:{name}"] = res

        return results

    def _internal_clean(self, prefix: str):
        """Cleans up ORCA run files, and returns to previous PWD

        Arguments:
             prefix (str): the prefix for the orca run files
        """
        run_cmd = f"rm {prefix}*"
        subprocess.run(run_cmd, shell=True, check=True)
        os.chdir(self._pwd)

    def _property_calc(
        self, mol: Molecule, search_string: str, density_needed: bool, tmp: str, **params
    ) -> Any:
        """Run an ORCA calculation and look up a property

        Arguments:
            mol (Molecule): molecule to run calculation on
            search_string (str): see _read_property_file
            density_needed (bool): if True, density explicitly
                   calculated - note this is not necessary for
                   HF/DFT calcs
            tmp (str): path to the scratch directory

        Returns:
            property value corresponding to search_string, if
            it exists

        Raises:
            FailedCalculation
        """
        if density_needed:
            if "density" not in params:
                if "mp2" in mol.method:
                    params["density"] = "unrelaxed"
                else:
                    params["density"] = "linearized"
        name = params.get("jobname", "energy")
        prefix = self.initialise(mol, tmp=tmp, name=name, **params)
        self._run_orca(prefix)
        results = self._read_property_file(prefix, [search_string])
        self._internal_clean(prefix)
        if results[search_string] is None:
            raise FailedCalculation
        return results[search_string]

    @available
    def energy(self, mol, tmp="", **params):
        search_string = "Calculation_Info:Total Energy"
        result = self._property_calc(mol, search_string, False, tmp, **params)
        return float(result)

    @available
    def jk_error(self, mol, tmp="", **params):
        fit_type = params.get("fit_type", "jk")
        cmd = params.get("command_line", "")
        if "RIJONX" in cmd:
            cmd = cmd.replace("RIJONX", "")
        if "RIJK" in cmd:
            cmd = cmd.replace("RIJK", "")

        if 'energy' not in mol._references:
            params["command_line"] = cmd
            mol.add_reference('energy', self.energy(mol, tmp=tmp, **params))

        if fit_type == "jk":
            cmd += " RIJK"
        else:
            cmd += " RIJONX"

        params["command_line"] = cmd
        mol.add_result('energy', self.energy(mol, tmp=tmp, **params))
        return mol.get_delta('energy')

    @available
    def dipole(self, mol, tmp="", **params):
        if "elprop" not in params:
            params["elprop"] = ["dipole\ttrue"]
        search_string = self._density_prefix(mol.method)
        if not search_string:
            search_string = "scf"
        search_string = search_string.upper() + "_Electric_Properties:Total Dipole moment"
        return self._property_calc(mol, search_string, True, tmp, **params)

    @available
    def quadrupole(self, mol, tmp="", **params):
        if "elprop" not in params:
            params["elprop"] = ["quadrupole\ttrue"]
        search_string = self._density_prefix(mol.method)
        if not search_string:
            search_string = "scf"
        search_string = search_string.upper() + "_Electric_Properties:Total quadrupole moment"
        return self._property_calc(mol, search_string, True, tmp, **params)

    @available
    def polarizability(self, mol, tmp="", **params):
        if "elprop" not in params:
            params["elprop"] = ["polar\ttrue"]
        search_string = self._density_prefix(mol.method)
        if not search_string:
            search_string = "scf"
        search_string = search_string.upper() + "_Electric_Properties:Isotropic polarizability"
        return self._property_calc(mol, search_string, True, tmp, **params)
