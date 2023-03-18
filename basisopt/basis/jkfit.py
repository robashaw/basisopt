from typing import Any, Optional, Union

from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import InternalBasis, OptResult
from basisopt.molecule import Molecule, build_diatomic
from basisopt.opt import Strategy, optimize
from basisopt.opt.reduce import ReduceStrategy
from basisopt.util import bo_logger

from .basis import Basis


class JKFitBasis(Basis):
    """Object for preparation and optimization of a an auxiliary basis set
    for a single atom, for the fitting of the Coulomb (and optionally Exchange)
    integrals in a Hartree-Fock or DFT calculation.

    Attributes:
         basis_type (str): "jfit/jkfit" for coulomb vs coulomb+exchange fitting
    """

    def __init__(
        self,
        name: str = 'H',
        charge: int = 0,
        mult=1,
        mol: Optional[Molecule] = None,
        jonly: bool = False,
    ):
        super().__init__()
        self.name = name
        if mol:
            self._molecule = mol
        else:
            mol_str = f"{name.title()}H,1.3"
            self._molecule = build_diatomic(mol_str, charge=charge, mult=mult)

        self.basis_type = "jkfit"
        if jonly:
            self.basis_type = "jfit"
        self._done_setup = False
        self.strategy = None

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of JKFitBasis"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["name"] = self.name
        d["basis_type"] = self.basis_type

        if isinstance(self.strategy, Strategy):
            d["strategy"] = self.strategy
            d["done_setup"] = self._done_setup
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates JKFitBasis from MSONable dictionary"""
        basis = Basis.from_dict(d)
        mol = basis._molecule
        charge = basis._molecule.charge
        mult = basis._molecule.multiplicity
        instance = cls(name=d.get("name", mol.name), charge=charge, mult=mult, mol=mol)
        instance.results = basis.results
        instance.opt_results = basis.opt_results
        instance._tests = basis._tests
        instance._molecule = basis._molecule
        instance.basis_type = d.get("basis_type", "jkfit")
        instance.strategy = d.get("strategy", None)
        if instance.strategy:
            instance._done_setup = d.get("done_setup", False)
        return instance

    def get_basis(self) -> InternalBasis:
        """Returns jfit or jkfit basis, depending on
        basis_type attribute
        """
        if self.basis_type == 'jfit':
            return self._molecule.jbasis
        return self._molecule.jkbasis

    def setup(
        self,
        basis: InternalBasis,
        guess: Optional[Union[str, InternalBasis]] = None,
        config: Optional[list[int]] = None,
        method: str = 'rhf',
        params: dict[str, Any] = {},
    ):
        """Sets up the basis ready for optimization. Must be called before optimize is called

        Arguments:
             basis (InternalBasis): the orbital basis to use
             guess (str or InternalBasis): the starting guess for jkfit basis
             config (list[int] or None): the desired number of functions per
                  shell in angular momentum order; if None, will just optimize
                  the whole starting guess
             method (str): the calculation method to use
             params (dict): parameters to pass to the backend

        Sets:
             self.strategy
             self.config
             self._done_setup - cannot call optimize until this flag is True
        """
        if guess:
            if isinstance(guess, str):
                starting_basis = fetch_basis(guess, self._molecule.unique_atoms())
            else:
                starting_basis = guess.copy()
        else:
            bo_logger.error("No basis guess given")
            return

        if config:
            self.strategy = ReduceStrategy(
                starting_basis,
                eval_type='jk_error',
                method=method,
                target=1.0,
                shell_mins=config,
                max_l=-1,
                reopt_all=False,
                **params,
            )
        else:
            self.strategy = Strategy(eval_type='jk_error')

        self.strategy.basis_type = self.basis_type
        self.strategy.orbital_basis = basis
        self._molecule.basis = basis
        self._molecule.method = method
        if self.basis_type == 'jfit':
            self._molecule.jbasis = starting_basis
        else:
            self._molecule.jkbasis = starting_basis
        self._done_setup = True

    def optimize(self, algorithm: str = 'Nelder-Mead', params: dict[str, Any] = {}) -> OptResult:
        """Runs the basis optimization

        Arguments:
             algorithm (str): optimization algorithm to use, see scipy.optimize for options
             params (dict): dictionary of parameters to pass to the backend -
             see the relevant Wrapper object for options
        """
        if self._done_setup:
            self.opt_results = optimize(
                self._molecule,
                element=self.name,
                algorithm=algorithm,
                strategy=self.strategy,
                **params,
            )
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results


def jkfit_collection(
    element: str,
    starting_guess: Union[str, InternalBasis],
    basis_pairs: list[tuple[InternalBasis, Optional[list[int]]]] = [],
    charge: int = 0,
    mult: int = 1,
    mol: Optional[Molecule] = None,
    jonly: bool = False,
    method: str = 'rhf',
    algorithm: str = 'Nelder-Mead',
    opt_params: dict[str, Any] = {},
    params: dict[str, Any] = {},
) -> list[JKFitBasis]:
    """Optimizes a collection of JKFit basis sets, in the style of cc-pVnZ basis sets,
    i.e. V5Z -> VQZ -> VTZ, by reducing the fitting set size and reoptimizing at each step.

    Arguments:
        element (str): the atomic element being optimized
        starting_guess (str or InternalBasis): the initial J(K)Fit basis to use
        basis_pairs (list of tuples): the orbital basis/configuration pairs, in order,
                e.g. [(v5z, None), (vqz, [10, 5, 3, 2, 1, 0]), (vtz, [10, 5, 2, 1, 0, 0])]
        charge (int), mult (int): the charge and multiplicity of the element-hydride
        mol (Molecule, optional): used instead of ElementH, overrides charge/mult
        jonly (bool): if True, only fit the Coulomb integrals, not the exchange
        method (str): computational method to use
        algorithm (str): scipy.optimize algorithm to use
        opt_params (dict): parameters to pass to the optimizer
        params (dict): parameters to pass to the backend

    Returns:
        a list of optimized JKFitBasis objects corresponding to the order of basis_pairs
    """
    results = []
    guess = starting_guess
    for basis, config in basis_pairs:
        new_jk = JKFitBasis(name=element, charge=charge, mult=mult, mol=mol, jonly=jonly)
        new_jk.setup(basis, guess=guess, config=config, method=method, params=params)
        _ = new_jk.optimize(algorithm=algorithm, params=opt_params)
        results.append(new_jk.copy())
        if jonly:
            guess = new_jk._molecule.jbasis
        else:
            guess = new_jk._molecule.jkbasis
    return results
