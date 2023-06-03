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
            'rks': ['energy'],
            'uks': ['energy'],
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

    def _convert_params(self, method: str, **params) -> str:
        """Helper function to pluck out the relevant
        parameters for a given method.
        Returns an empty string if the params are not found.
        """
        method_search = method + '-params'
        param_options = params.get(method_search, '')
        return param_options

    def _command_string(self, method: str, **params) -> str:
        """Helper function to turn an internal method
        name into a Molpro command line
        """
        # Extract the relevant parameters for this method
        method_options = self._convert_params(method, **params)
        # Post-HF calculations need to specify the HF part too
        # hence check for any user-supplied params for the HF part.
        if method in ['mp2', 'ccsd', 'ccsd(t)']:
            ref_options = self._convert_params('hf', **params)
            command = f"{{hf{ref_options}}}\n{{{method}{method_options}}}"
        elif method in ['rmp2', 'uccsd', 'uccsd(t)']:
            ref_options = self._convert_params('rhf', **params)
            command = f"{{rhf{ref_options}}}\n{{{method}{method_options}}}"
        elif method in ['ump2']:
            ref_options = self._convert_params('uhf', **params)
            command = f"{{uhf{ref_options}}}\n{{{method}{method_options}}}"
        # For DFT based methods, the functional must be specified separately
        # to other params
        elif method in ['rks', 'uks']:
            if "functional" in params:
                xcfun = params["functional"]
                command = f"{{{method},{xcfun}{method_options}}}"
            else:
                raise KeyError("DFT functional not specified")
        else:
            command = f"{{{method}{method_options}}}"
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
        g_param_str = ""
        glo_params = params.get("global-params", '')
        if glo_params:
            g_param_str = glo_params + "\n"
        cmd = self._command_string(m.method, **params)
        mol = self.convert_molecule(m)
        basis = self._convert_basis(m.basis)

        # Assemble into an input string and pass to the Project
        molpro_input = g_param_str + mol + basis + cmd
        print(molpro_input)
        proj.write_input(molpro_input)

    def _get_energy(self, proj: Project, meth: str) -> float:
        """Helper function to retrieve the energy from
        Molpro after a calculation"""
        if meth in ['hf', 'rhf', 'uhf', 'rks', 'uks']:
            energy = proj.energies()[0]
        elif meth in ['mp2']:
            energy = proj.energies()[-1]
        elif meth in ['rmp2', 'ump2', 'ccsd']:
            energy = proj.energies(method=f"{meth.upper()}")[-1]
        elif meth in ['uccsd', 'uccsd(t)']:
            energy = proj.energy(method=f"RHF-{meth.upper()}")
        else:
            energy = proj.energy(method=f"{meth.upper()}")
        return energy

    @available
    def energy(self, mol, tmp="", **params):
        name = "energy"
        proj_name = f"{mol.name}-{mol.method}-" + name
        p = Project(proj_name, location=tmp)
        self.initialise(p, mol, tmp=tmp, **params)
        p.run(wait=True)
        if p.errors():
            raise FailedCalculation
        # Attempt to catch race cases where wait=True doesn't seem to be sufficient
        p.wait()
        # TODO use a helper function to extract the energy for a particular method
        energy = self._get_energy(p, mol.method)
        #        if mol.method == "hf":
        #            energy = p.energy()
        #        else:
        #            energy = p.energy(method=f"{mol.method.upper()}")
        p.clean()
        return energy
