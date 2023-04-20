# Wrappers for Molpro functionality using pymolpro (https://github.com/molpro/pymolpro)
from pymolpro import Project

from basisopt.bse_wrapper import internal_basis_converter
from basisopt.containers import InternalBasis
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.wrappers.wrapper import Wrapper, available


class MolproWrapper(Wrapper):
    """Wrapper for Molpro using pymolpro"""

    def __init__(self):
        super().__init__(name='Molpro')
        self._method_strings = {
            'hf': ['energy'],
            'rhf': ['energy'],
            'uhf': ['energy'],
        }

    def convert_molecule(self, m: Molecule) -> str:
        """Convert an internal Molecule object
        to a Molpro geometry section and set charge/multiplicity
        """
        molstring = "geomtyp=xyz\n"
        molstring += "geom={\n"
        for i in range(m.natoms()):
            molstring += m.get_line(i) + "\n"
        molstring += "}\n"
        molstring += f"set,charge={m.charge}\n"
        spin = m.multiplicity - 1
        molstring += f"set,spin={spin}\n"
        return molstring

    def _command_string(self, method: str, **params) -> str:
        """Helper function to turn an internal method
        name into a Molpro command line
        """
        command = f"{method}"
        return command + "\n"

    def _convert_basis(self, basis: InternalBasis) -> str:
        """Converts an InternalBasis to the basis string for Molpro
        May be updated in future to enable auxiliary fitting sets

        Arguments:
            basis (InternalBasis): basis set to convert

        Returns:
            Molpro basis block string
        """
        molpro_basis = internal_basis_converter(basis, fmt="molpro").split("\n")
        basis_str = ""
        for line in molpro_basis[1:-1]:
            basis_str += line + "\n"
        return basis_str

    def initialise(self, m: Molecule, name: str = "", tmp: str = "", **params) -> Project:
        """Initialises pymolpro before each calculation
        - creates a pymolpro Project
        - converts molecule
        - sets options from globals
        - converts basis set
        """
        # Create a Project
        proj_name = f"{m.name}-{m.method}-" + name
        p = Project(proj_name, location=tmp)

        # Handle options, molecule, basis
        # TODO - add params
        cmd = self._command_string(m.method)
        mol = self.convert_molecule(m)
        basis = self._convert_basis(m.basis)

        # Assemble into an input string and pass to the Project
        molpro_input = mol + basis + cmd
        p.write_input(molpro_input)
        return p

    #    def _get_properties(
    #        self, mol: Molecule, name: str = "prop", properties: list[str] = [], tmp: str = "", **params
    #    ) -> dict[str, Any]:
    #        """Helper function to run the calculation and
    #        retrieve a property value other than energy from Molpro.
    #        """
    #        self.initialise(mol, name=name, tmp: str = "", **params)
    #        p.run(wait=True)
    #        if p.errors():
    #            raise FailedCalculation

    # Retrieve the results

    @available
    def energy(self, mol, tmp="", **params):
        p = self.initialise(mol, name="energy", tmp=tmp, **params)
        p.run(wait=True)
        if p.errors():
            raise FailedCalculation
        energy = p.energy()
        p.clean()
        return energy
