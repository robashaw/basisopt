# base test types
from basisopt.molecule import Molecule
from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import Result
from basisopt import api
from basisopt.exceptions import EmptyCalculation

class Test(Result):
    """Abstract Test class, a type of Result, for a way of testing a basis set.
       see e.g. PropertyTest and DunhamTest
    
       Attributes:
            reference (var): reference value of some kind to compare test result to
            molecule: Molecule object to perform test with
    
       Must implement in children:
            calculate(self, method, basis, params={})
    """
    def __init__(self, name, reference=None, mol=None, xyz_file=None, charge=0, mult=1):
        Result.__init__(self, name)
        self.reference = reference
        self.molecule = None
        
        if mol is not None:
            self.molecule = mol
        elif xyz_file is not None:
            self.set_molecule_from_xyz(xyz_file, charge=charge, mult=mult)
        
    def set_molecule_from_xyz(self, xyz, charge=0, mult=1):
        """Creates Molecule from xyz file"""
        self.molecule = Molecule(self.name, charge=charge, mult=mult)
        self.molecule.from_xyz(xyz)
        
    def calculate_reference(self, method, basis=None, basis_name='cc-pvqz', params={}):
        """Calculates reference value for the test, should not need to be overridden
        
           Attributes:
                method (str): method to run, e.g. 'rhf', 'mp2'
                basis: internal basis dictionary 
                OR if basis is None:
                    basis_name (str): calculate using basis with this name from BSE
                params (dict): parameters to pass to the backend Wrapper
        """
        if basis is None:
            basis = fetch_basis(basis_name, self.molecule.unique_atoms())
        self.reference = self.calculate(method, basis, params=params)
        
    def calculate(self, method, basis, params={}):
        """Interface to run the test. Should archive and return the results
           of the test.
           
           Attributes:
                method (str): method to run, e.g. 'rhf', 'mp2'
                basis: internal basis dictionary
                params (dict): parameters to pass to the backend Wrapper
        """
        raise NotImplementedException
    
class PropertyTest(Test):
    """Simplest implementation of Test, calculating some property, e.g. energy
    
       Additional attributes:
            eval_type (str): property to evaluate, e.g. 'energy', 'dipole'
    """
    def __init__(self, name, prop='energy', mol=None, xyz_file=None, charge=0, mult=1):
        Test.__init__(self, name, mol=mol, xyz_file=xyz_file, charge=charge, mult=mult)
        self._eval_type = ''
        self.eval_type  = prop
    
    @property
    def eval_type(self):
        return self._eval_type
    
    @eval_type.setter
    def eval_type(self, name):
        wrapper = api.get_backend()
        if name in wrapper.all_available():
            self._eval_type = name
        else:
            raise PropertyNotAvailable(name)
    
    def calculate(self, method, basis, params={}):
        if self.molecule is None:
            raise EmptyCalculation
        # run calculation
        self.molecule.basis = basis
        self.molecule.method = method
        success = api.run_calculation(evaluate=self.eval_type, mol=self.molecule, params=params)
        
        # retrieve result, archive _and_ return
        wrapper = api.get_backend()
        value = wrapper.get_value(self.eval_type)
        self.add_data(self.name+self.eval_type, value)
        return value
        
        