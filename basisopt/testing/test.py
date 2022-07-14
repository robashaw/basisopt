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
        super(Test, self).__init__(self, name)
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
        
    def as_dict(self):
        d = super(Test, self).as_dict()        
        d["@module"] = type(self).__module__
        d["@class"]  = type(self).__name__
        d["reference"] = self.reference
        if self.molecule:
            d["molecule"] = self.molecule.as_dict()
        return d
    
    @classmethod
    def from_dict(cls, d):
        name = d.get("name", "Empty")
        ref  = d.get("reference", None)
        molecule = d.get("molecule", None)
        if molecule:
            molecule = Molecule.from_dict(molecule)
        instance = cls(name, reference=ref, mol=molecule)
        result = Result.from_dict(d)
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
        
        # retrieve result, archive _and_ return
        wrapper = api.get_backend()
        value = wrapper.get_value(self.eval_type)
        self.add_data(self.name+"_"+self.eval_type, value)
        return value
        
    def as_dict(self):
        d = super(PropertyTest, self).as_dict()        
        d["@module"] = type(self).__module__
        d["@class"]  = type(self).__name__
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
        
        