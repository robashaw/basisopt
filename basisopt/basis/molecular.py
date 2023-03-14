import pickle
from typing import Any, Callable, Optional, Union

import numpy as np

from basisopt import api
from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import (
    InternalBasis,
    OptCollection,
    Result,
    basis_to_dict,
    dict_to_basis,
)
from basisopt.exceptions import DataNotFound, EmptyBasis
from basisopt.molecule import Molecule
from basisopt.opt.optimizers import collective_optimize
from basisopt.opt.strategies import Strategy
from basisopt.util import bo_logger

from .atomic import AtomicBasis
from .basis import Basis


class MolecularBasis(Basis):
    """Object for preparation and optimization of a basis set for
    multiple atoms across one or more Molecules.

    Attributes:
         basis (dict): internal basis used for all molecules

    Private Attributes:
         _molecules (dict): dictionary of Molecule objects
         _atoms (set): unique atoms across all molecules
         _atomic_bases (dict): dictionary of AtomicBasis objects
             for each atom in _atoms
         _done_setup (bool): if True, setup has been called
    """

    def __init__(self, name: str = 'Empty', molecules: list[Molecule] = []):
        super().__init__()
        self.name = name
        self.basis = {}
        self._molecules = {}
        self._atoms = set()
        self._atomic_bases = {}
        self._done_setup = False
        for m in molecules:
            self.add_molecule(m)

    def save(self, filename: str):
        """Pickles the MolecularBasis object into a binary file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
        bo_logger.info("Dumped object of type %s to %s", type(self), filename)

    def as_dict(self) -> dict[str, Any]:
        """Returns as MSONable dictionary"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["basis"] = basis_to_dict(self.basis)
        d["atoms"] = self._atoms
        d["atomic_bases"] = {k: ab.as_dict() for k, ab in self._atomic_bases.items()}
        d["done_setup"] = self._done_setup
        d["molecules"] = {k: m.as_dict() for k, m in self._molecules.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates a MolecularBasis from an MSONable dictionary"""
        basis = Basis.from_dict(d)
        instance = cls(name=basis.name)
        instance.results = basis.results
        instance.opt_results = basis.opt_results
        instance._tests = basis._tests
        instance.basis = dict_to_basis(d.get("basis", {}))
        instance._atoms = d.get("atoms", set())
        instance._done_setup = d.get("done_setup", False)
        instance._atomic_bases = d.get("atomic_bases", {})
        instance._molecules = d.get("molecules", {})
        return instance

    def add_molecule(self, molecule: Molecule):
        """Adds a Molecule object to the optimization pool"""
        if molecule.name in self._molecules:
            bo_logger.warning("Molecule with name %s being overwritten", molecule.name)
        self._molecules[molecule.name] = molecule
        for atom in molecule.unique_atoms():
            self._atoms.add(atom.lower())

    def get_molecule(self, name: str) -> Molecule:
        """Returns a Molecule with the given name, if it exists, otherwise None"""
        if name in self._molecules:
            return self._molecules[name]
        bo_logger.warning("No molecule with name %s", name)
        return None

    def get_basis(self) -> InternalBasis:
        """Returns the basis set used for all molecules"""
        return self.basis

    def get_atomic_basis(self, atom: str) -> AtomicBasis:
        """Returns the AtomicBasis object for a given atom, if it exists,
        otherwise None
        """
        if atom in self._atomic_bases:
            return self._atomic_bases[atom]
        return None

    def unique_atoms(self) -> list[str]:
        """Returns list of unique atoms across all molecules"""
        return list(self._atoms)

    def molecules(self) -> list[Molecule]:
        """Returns a list of all the Molecule objects"""
        return list(self._molecules.values())

    def run_test(
        self,
        name: str,
        params: dict[str, Any] = {},
        reference_basis: Optional[Union[str, InternalBasis]] = None,
        do_print: bool = True,
    ) -> dict[str, Any]:
        """Runs a single test with a given name across all molecules

        Arguments:
             name (str): name of the test
             params (dict): parameters for backend
             reference_basis (str or dict): either string name for basis to fetch
                 from the BSE, or an internal basis dictionary, or None
             do_print (bool): if True, test results will be printed to Logger

        Returns:
             Dicionary of results for each test, indexed by molecule name
        """
        t = self.get_test(name)
        results = {}
        if t is None:
            bo_logger.warning("No test with name %s", name)
        else:
            try:
                child = self.results.get_child(name)
            except DataNotFound:
                new_result = Result(name=name)
                self.results.add_child(new_result)
                child = new_result

                # calculate reference values
                bo_logger.info("Calculating reference values for test %s", name)
                str_basis = isinstance(reference_basis, str)
                for m in self.molecules():
                    t.molecule = m
                    if str_basis:
                        t.calculate_reference(m.method, basis_name=reference_basis, params=params)
                    else:
                        t.calculate_reference(m.method, basis=reference_basis, params=params)
                    child.add_data(f"{m.name}_ref", t.reference)

            for m in self.molecules():
                t.result = t.calculate(m.method, self.basis, params=params)
                child.add_data(m.name, t.result)
                results[m.name] = t.result
                if do_print:
                    bo_logger.info("%s: %s", m.name, str(t.result))
        return results

    def run_all_tests(
        self,
        params: dict[str, Any] = {},
        reference_basis: Optional[Union[str, InternalBasis]] = None,
    ) -> None:
        """Runs all of the tests across all molecules, and prints the results to logger

        Arguments:
             params (dict): paramerters to pass to the backend
             reference_basis (str or dict): either string name for basis to fetch
                 from the BSE, or an internal basis dictionary, or None
        """
        results = {}
        for t in self._tests:
            results[t.name] = self.run_test(
                t.name, params=params, reference_basis=reference_basis, do_print=False
            )
        # print results
        header = "Molecule"
        for t in self._tests:
            header += f"\t{t.name}"
        bo_logger.info(header)
        for m in self.molecules():
            res_string = f"{m.name}"
            for v in results.values():
                res_string += f"\t{v[m.name]}"
            bo_logger.info(res_string)

    def setup(
        self,
        method: str = 'ccsd(t)',
        quality: str = 'dz',
        strategy: Strategy = Strategy(),
        reference: str = 'cc-pvqz',
        params: dict[str, Any] = {},
    ):
        """Sets up the basis ready for optimization by creating AtomicBasis objects for each unique
        atom in the set, and calling setup for those - see the signature of AtomicBasis.setup for
        explanation.
        """
        if len(self._atoms) == 0:
            raise EmptyBasis

        for m in self.molecules():
            m.method = method

        self._atomic_bases = {}
        for atom in self._atoms:
            self._atomic_bases[atom] = AtomicBasis(atom)
            bo_logger.info("Doing setup for atom %s", atom)
            self._atomic_bases[atom].setup(
                method=method,
                quality=quality,
                strategy=strategy,
                reference=('dummy', 0.0),
                params=params,
            )
        self.basis = {k: v.get_basis()[k] for k, v in self._atomic_bases.items()}
        if reference is not None:
            if api.which_backend() == 'Empty':
                bo_logger.warning("No backend currently set, can't compute reference value")
            else:
                ref_basis = fetch_basis(reference, self.unique_atoms())
                for m in self.molecules():
                    bo_logger.info(
                        "Calculating reference value for molecule %s using %s and %s/%s",
                        m.name,
                        api.which_backend(),
                        method,
                        reference,
                    )
                    m.basis = ref_basis
                    success = api.run_calculation(evaluate=strategy.eval_type, mol=m, params=params)
                    if success != 0:
                        bo_logger.warning("Reference calculation failed")
                        value = 0.0
                    else:
                        value = api.get_backend().get_value(strategy.eval_type)
                    m.add_reference(strategy.eval_type, value)
                    bo_logger.info("Reference value set to %f", value)

        self._done_setup = True
        bo_logger.info("Molecular basis setup complete")

    def optimize(
        self,
        algorithm: str = 'Nelder-Mead',
        params: dict[str, Any] = {},
        reg: Callable[[np.ndarray], float] = lambda x: 0,
        npass: int = 1,
        parallel: bool = False,
    ) -> OptCollection:
        """Calls collective optimize to optimize all the atomic basis sets in this basis

        Arguments:
             algorithm (str): name of scipy.optimize algorithm to use
             params (dict): parameters to pass to scipy.optimize
             reg (callable): regularization to use
             npass (int): number of optimization passes to do
             parallel (bool): if True, molecular calculations will be distributed in parallel

         Returns:
             dictionary of scipy.optimize result objects, indexed by atom
        """
        if self._done_setup:
            opt_data = [
                (k, algorithm, v.strategy, reg, params) for k, v in self._atomic_bases.items()
            ]
            self.opt_results = collective_optimize(
                self._molecules.values(),
                self.basis,
                opt_data=opt_data,
                npass=npass,
                parallel=parallel,
            )
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results
