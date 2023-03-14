# Wrappers for psi4 functionality
from typing import Any

import psi4

from basisopt.bse_wrapper import internal_basis_converter
from basisopt.exceptions import EmptyCalculation, PropertyNotAvailable
from basisopt.molecule import Molecule
from basisopt.wrappers.wrapper import Wrapper, available


class Psi4Wrapper(Wrapper):
    """Wrapper for Psi4"""

    def __init__(self):
        super().__init__(name='Psi4')
        self._method_strings = {
            'scf': ['energy', 'dipole', 'quadrupole'],
            'hf': ['energy', 'dipole', 'quadrupole'],
            'dft': ['energy', 'dipole', 'quadrupole'],
            'mp2': ['energy', 'dipole', 'quadrupole'],
            'mp3': ['energy'],
            'mp4': ['energy'],
            'lccd': ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'lccsd': ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cc2': ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cc3': ['energy'],
            'ccsd': ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'ccsd(t)': ['energy', 'dipole', 'quadrupole', 'polarizability'],
            'cisd': ['energy', 'trans_dipole', 'trans_quadrupole'],
            'ras-ci': ['energy'],
            'casscf': ['energy', 'trans_dipole', 'trans_quadrupole'],
            'rasscf': ['energy', 'trans_dipole', 'trans_quadrupole'],
            'adc(1)': ['energy', 'dipole', 'trans_dipole'],
            'adc(2)': ['energy', 'dipole', 'trans_dipole'],
            'adc(3)': ['energy', 'dipole', 'trans_dipole'],
        }
        self._restricted_options = ["functional"]

    def convert_molecule(self, m: Molecule) -> psi4.core.Molecule:
        """Convert an internal Molecule object
        to a Psi4 Molecule object
        """
        molstring = ""
        for i in range(m.natoms()):
            molstring += m.get_line(i) + "\n"
        return psi4.geometry(molstring)

    def _property_prefix(self, method: str) -> str:
        """Helper function to lookup properties from
        psi4.properties
        """
        m = method.lower()
        if m in ['scf', 'hf']:
            return 'SCF'
        elif m in ['cisd']:
            return 'CI'
        return method.upper()

    def _command_string(self, method: str, **params) -> str:
        """Helper function to turn an internal method
        name into a psi4 run string
        """
        if method == "dft":
            if "functional" in params:
                command = params["functional"]
            else:
                raise KeyError("DFT functional not specified")
        else:
            command = method
        return command

    def initialise(self, m: Molecule, name: str = "", tmp: str = "", **params):
        """Initialises Psi4 before each calculation
        - sets output file
        - converts molecule
        - sets options from globals and params
        - converts basis set (TODO: handle jkfit)
        """
        # create output file
        outfile = tmp + f"{m.name}-{m.method}-" + name + ".out"
        psi4.core.set_output_file(outfile, False)

        # create the molecule
        mol = self.convert_molecule(m)
        mol.set_molecular_charge(m.charge)
        mol.set_multiplicity(m.multiplicity)

        options = {k: v for k, v in self._globals.items() if k not in self._restricted_options}
        for k, v in params.items():
            if k not in self._restricted_options:
                options[k] = v

        # logic to check global options
        # TODO: expand option handling
        if "memory" in options:
            psi4.set_memory(self._globals["memory"])
            del options["memory"]
        psi4.set_options(options)

        # set basis
        g94_basis = internal_basis_converter(m.basis, fmt="psi4")
        psi4.basis_helper(g94_basis)

    def clean(self):
        """Cleans up calculation"""
        psi4.core.clean()

    def _get_properties(
        self, mol: Molecule, name: str = "prop", properties: list[str] = [], tmp: str = "", **params
    ) -> dict[str, Any]:
        """Helper function to retrieve a property value from Psi4
        after a calculation.
        """
        ptype = self._property_prefix(mol.method)
        if ptype == "DFT":
            func = params['functional']
            ptype = func.upper()
        strings = [ptype + " " + p.upper() for p in properties]

        if len(strings) == 0:
            raise EmptyCalculation

        self.initialise(mol, name=name, tmp=tmp, **params)
        runstring = self._command_string(mol.method, **params)
        _, wfn = psi4.properties(runstring, return_wfn=True, properties=properties)

        results = {}
        for p, s in zip(properties, strings):
            if s in wfn.variables():
                results[p] = wfn.variable(s)
            else:
                raise PropertyNotAvailable(p)

        return results

    @available
    def energy(self, mol, tmp="", **params):
        self.initialise(mol, name="energy", tmp=tmp, **params)
        runstring = self._command_string(mol.method, **params)
        return psi4.energy(runstring)

    @available
    def dipole(self, mol, tmp="", **params):
        results = self._get_properties(mol, name="dipole", properties=['dipole'], tmp=tmp, **params)
        return results['dipole']

    @available
    def quadrupole(self, mol, tmp="", **params):
        results = self._get_properties(
            mol, name="quadrupole", properties=['quadrupole'], tmp=tmp, **params
        )
        return results['quadrupole']

    @available
    def polarizability(self, mol, tmp="", **params):
        results = self._get_properties(
            mol, name="polar", properties=['polarizability'], tmp=tmp, **params
        )
        return results['polarizability']
