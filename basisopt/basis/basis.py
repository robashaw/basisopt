import copy
import logging
import pickle
import numpy as np
from monty.json import MSONable
from basisopt.containers import Result, Shell
from basisopt.util import bo_logger
from basisopt.molecule import Molecule
from basisopt import data

def uncontract_shell(shell):
    """Converts a Shell into an uncontracted Shell
       (overwrites any existing contraction coefs)
    """
    shell.coefs = []
    n = shell.exps.size
    for ix in range(n):
        c = np.zeros(n)
        c[ix] = 1.0
        shell.coefs.append(c)

def uncontract(basis, elements=None):
    """Uncontracts all shells in a basis for the elements specified
       (does not overwrite the old basis).
    
       Arguments:
            basis (dict): the basis dictionary to be uncontracted
            elements (list): list of atomic symbols 
    
       Returns:
            a new basis dictionary with uncontracted shells
    """
    if elements is None:
        elements = basis.keys() # do all
    new_basis = copy.copy(basis)
    for el in elements:
        if el in new_basis:
            el_basis = new_basis[el]
            for s in el_basis:
                uncontract_shell(s)
    return new_basis
    
def even_temper_expansion(params):
    """Forms a basis for an element from even tempered expansion parameters
    
       Arguments:
            params (list): list of tuples corresponding to shells
            e.g. [(c_s, x_s, n_s), (c_p, x_p, n_p), ...] where each shell
            is expanded as c_l * (x_l**k) for k=0,...,n_l
    
       Returns:
            list of Shell objects for the expansion
    """
    el_basis = []
    for ix, (c, x, n) in enumerate(params):
        new_shell = Shell()
        new_shell.l = data.INV_AM_DICT[ix]
        new_shell.exps = np.array([c*(x**p) for p in range(n)])
        uncontract_shell(new_shell)
        el_basis.append(new_shell)
    return el_basis
    
def fix_ratio(exps, ratio=1.4):
    """Returns a sorted numpy array of exponents
       where x_{i+1}/x_i >= ratio
    """
    new_exps = np.sort(exps)
    for i in range(exps.size-1):
        if new_exps[i+1]/new_exps[i] < ratio:
            new_exps[i+1] = new_exps[i]*ratio
    return new_exps
    
class Basis(MSONable):
    """Abstract parent class object representing a basis type
       All basis types must inherit from here to work, see e.g. AtomicBasis, MolecularBasis
    
       Attributes:
            results: a Result object where any results (e.g. calculations, optimizations, ...)
            can be archived
    
       Private attributes:
            _tests (list): a list of Test objects that can be run and collated together, with results
            going into the results attribute
    """
    def __init__(self):
        self.results = Result()
        self.opt_results = None
        self._tests = []
        self._molecule = None
            
    def save(self, filename):
        """Pickles the Basis object into a binary file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
        bo_logger.info("Dumped object of type %s to %s", type(self), filename)
    
    def load(self, filename):
        """Loads and returns a Basis object from a binary file pickle"""
        with open(filename, 'rb') as f:
            pkl_data = pickle.load(f)
            f.close()
        bo_logger.info("Loaded object of type %s from %s", type(pkl_data), filename)
        return pkl_data
        
    def get_basis(self):
        return self._molecule.basis
    
    def register_test(self, test):
        """Add a Test object to the set of tests"""
        self._tests.append(test)
        
    def get_test(self, name):
        """Retrieve a Test with a given name if it exists"""
        for t in self._tests:
            if t.name == name:
                return t
        return None
        
    def run_test(self, name, params={}):
        """Runs a test with the given name, printing result"""
        t = self.get_test(name)
        if t is None:
            bo_logger.warning("No test with name %s", name)
        else:
            t.result = t.calculate(self._molecule.method, self._molecule.basis, params=params)
            bo_logger.info("Test %s: %s", name, t.result)
    
    def run_all_tests(self, params={}):
        """Runs all the tests in basis, printing results"""
        for t in self._tests:
            t.result = t.calculate(self._molecule.method, self._molecule.basis, params=params)
            bo_logger.info("Test %s: %s", name, t.result)
            
    def optimize(self, algorithm='Nelder-Mead', params={}):
        """All basis objects should implement an optimize method with this signature"""
        raise NotImplementedException
        
    def as_dict(self):
        d = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "results": self.results.as_dict(),
            "opt_results": self.opt_results,
            "tests": [t.as_dict() for t in self._tests],
            "molecule": self._molecule.as_dict()
        }
        return d
       
    @classmethod 
    def from_dict(cls, d):
        instance = cls()
        instance.results = d.get("results", Result())
        instance.opt_results = d.get("opt_results", None)
        molecule = d.get("molecule", None)
        if molecule:
            instance._molecule = Molecule.from_dict(molecule) 
        tests = d.get("tests", [])
        for t in tests:
            instance._tests.append(Test.from_dict(t))
        return instance