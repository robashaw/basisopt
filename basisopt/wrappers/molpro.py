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
            'mp2': ['energy'],
            'rmp2': ['energy'],
            'ump2': ['energy'],
            'ccsd': ['energy'],
            'ccsd(t)': ['energy'],
            'uccsd': ['energy'],
            'uccsd(t)': ['energy'],
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
        # Post-HF calculations need to specify the HF part too
        if method == "mp2":
            command = "hf\nmp2"
        elif method == "rmp2":
            command = "rhf\nrmp2"
        elif method == "ump2":
            command = "uhf\nump2"
        elif method[0:2] == "cc":
            command = f"hf\n{method}"
        elif method[0:3] == "ucc":
            command = f"rhf\n{method}"
        else:
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

    def initialise(self, proj: Project, m: Molecule, tmp: str = "", **params):
        """Initialises pymolpro before each calculation
        - creates a pymolpro Project
        - converts molecule
        - sets options from globals
        - converts basis set
        """

        # Handle options, molecule, basis
        # TODO - add params
        cmd = self._command_string(m.method)
        mol = self.convert_molecule(m)
        basis = self._convert_basis(m.basis)

        # Assemble into an input string and pass to the Project
        molpro_input = mol + basis + cmd
        proj.write_input(molpro_input)

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
        name = "energy"
        proj_name = f"{mol.name}-{mol.method}-" + name
        p = Project(proj_name, location=tmp)
        self.initialise(p, mol, tmp=tmp, **params)
        p.run(wait=True)
        if p.errors():
            raise FailedCalculation
        # TODO attempt to catch race cases where wait=True doesn't seem to be working
        p.wait()
        if mol.method == "hf":
            energy = p.energy()
        else:
            energy = p.energy(method=f"{mol.method.upper()}")
        p.clean()
        return energy
