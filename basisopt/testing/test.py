# base test types
from typing import Any, Optional

from basisopt import api
from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import InternalBasis, Result
from basisopt.exceptions import (
    EmptyCalculation,
    FailedCalculation,
    PropertyNotAvailable,
)
from basisopt.molecule import Molecule


class Test(Result):
    """Abstract Test class, a type of Result, for a way of testing a basis set.
    see e.g. PropertyTest and DunhamTest

    Attributes:
         reference (var): reference value of some kind to compare test result to
         molecule: Molecule object to perform test with

    Must implement in children:
         calculate(self, method, basis, params={})
    """

    def __init__(
        self,
        name: str,
        reference: Optional[Any] = None,
        mol: Optional[Molecule] = None,
        xyz_file: Optional[str] = None,
        charge: int = 0,
        mult: int = 1,
    ):
        super().__init__(name)
        self.reference = reference
        self.molecule = None

        if mol:
            self.molecule = mol
        elif xyz_file is not None:
            self.set_molecule_from_xyz(xyz_file, charge=charge, mult=mult)

    def set_molecule_from_xyz(self, xyz: str, charge: int = 0, mult: int = 1):
        """Creates Molecule from xyz file

        Arguments:
             xyz (str): the xyz file
             charge (int), mult (int): the charge and multiplicity of the molecule
        """
        self.molecule = Molecule(self.name, charge=charge, mult=mult)
        self.molecule.from_xyz(xyz)

    def calculate_reference(
        self,
        method: str,
        basis: Optional[InternalBasis] = None,
        basis_name: str = 'cc-pvqz',
        params: dict[str, Any] = {},
    ):
        """Calculates reference value for the test, should not need to be overridden

        Attributes:
             method (str): method to run, e.g. 'rhf', 'mp2'
             basis: internal basis dictionary
             basis_name (str): calculate using basis with this name from BSE,
                         if basis is None
             params (dict): parameters to pass to the backend Wrapper
        """
        if basis is None:
            basis = fetch_basis(basis_name, self.molecule.unique_atoms())
        self.reference = self.calculate(method, basis, params=params)

    def calculate(self, method: str, basis: InternalBasis, params: dict[str, Any] = {}):
        """Interface to run the test. Should archive and return the results
        of the test.

        Attributes:
             method (str): method to run, e.g. 'rhf', 'mp2'
             basis: internal basis dictionary
             params (dict): parameters to pass to the backend Wrapper
        """
        raise NotImplementedError

    def as_dict(self):
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["reference"] = self.reference
        if self.molecule:
            d["molecule"] = self.molecule.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        result = Result.from_dict(d)
        name = d.get("name", "Empty")
        ref = d.get("reference", None)
        molecule = d.get("molecule", None)
        instance = cls(name, reference=ref, mol=molecule)
        instance._data_keys = result._data_keys
        instance._data_values = result._data_values
        instance._children = result._children
        instance.depth = result.depth
        return instance


class PropertyTest(Test):
    """Simplest implementation of Test, calculating some property, e.g. energy

    Additional attributes:
         eval_type (str): property to evaluate, e.g. 'energy', 'dipole'
    """

    def __init__(
        self,
        name: str,
        prop: str = 'energy',
        mol: Optional[Molecule] = None,
        xyz_file: Optional[str] = None,
        charge: int = 0,
        mult: int = 1,
    ):
        super().__init__(name, mol=mol, xyz_file=xyz_file, charge=charge, mult=mult)
        self._eval_type = ''
        self.eval_type = prop

    @property
    def eval_type(self) -> str:
        return self._eval_type

    @eval_type.setter
    def eval_type(self, name: str):
        wrapper = api.get_backend()
        if name in wrapper.all_available():
            self._eval_type = name
        else:
            raise PropertyNotAvailable(name)

    def calculate(self, method: str, basis: InternalBasis, params: dict[str, Any] = {}) -> Any:
        """Calculates the test value

        Arguments:
             method (str): the method to use, e.g. 'scf'
             basis (dict): internal basis object
             params (dict): parameters to pass to backend

        Returns:
             the value from the calculation
        """
        if self.molecule is None:
            raise EmptyCalculation
        # run calculation
        self.molecule.basis = basis
        self.molecule.method = method
        success = api.run_calculation(evaluate=self.eval_type, mol=self.molecule, params=params)
        if success != 0:
            raise FailedCalculation

        # retrieve result, archive _and_ return
        wrapper = api.get_backend()
        value = wrapper.get_value(self.eval_type)
        self.add_data(self.name + "_" + self.eval_type, value)
        return value

    def as_dict(self):
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["eval_type"] = self.eval_type
        return d

    @classmethod
    def from_dict(cls, d):
        test = Test.from_dict(d)
        prop = d.get("eval_type", 'energy')
        instance = cls(test.name, prop=prop, mol=test.molecule)
        instance.reference = test.reference
        instance._data_keys = test._data_keys
        instance._data_values = test._data_values
        instance._children = test._children
        instance.depth = test.depth
        return instance
