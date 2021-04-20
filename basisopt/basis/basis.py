import pickle
from basisopt.containers import Result, Shell
from basisopt import data
import numpy as np
import copy

def uncontract_shell(shell):
    shell.coefs = []
    n = shell.exps.size
    for ix in range(n):
        c = np.zeros(n)
        c[ix] = 1.0
        shell.coefs.append(c)

def uncontract(basis, elements=None):
    if elements is None:
        elements = basis.keys() # do all
    new_basis = copy.copy(basis)
    for el in elements:
        el_basis = new_basis[el]
        for s in el_basis:
            uncontract_shell(s)
    return new_basis
    
def even_temper_expansion(params):
    el_basis = []
    for ix, (c, x, n) in enumerate(params):
        new_shell = Shell()
        new_shell.l = data.INV_AM_DICT[ix]
        new_shell.exps = np.array([c*(x**p) for p in range(n)])
        uncontract_shell(new_shell)
        el_basis.append(new_shell)
    return el_basis
    
def fix_ratio(exps, ratio=1.4):
    exps = np.sort(exps)
    for i in range(exps.size-1):
        if (exps[i+1]/exps[i] < ratio):
            exps[i+1] = exps[i]*ratio
    return exps
    
class Basis:
    def __init__(self):
        self.results = Result()
        self._tests = []
        self._molecule = None
            
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data
    
    def register_test(self, test):
        self._tests.append(test)
        
    def get_test(self, name):
        for t in self._tests:
            if t.name == name:
                return t
        return None
        
    def run_test(self, name, params={}):
        t = self.get_test(name)
        if t is None:
            print(f"No test with name {name}")
        else:
            t.result = t.calculate(self._molecule.basis, params=params)
            print(f"Test {name}: {t.result}")
    
    def run_all_tests(self, params={}):
        for t in self._tests:
            t.result = t.calculate(self._molecule.basis, params=params)
            print(f"Test {name}: {t.result}")
            
    def optimize(self, algorithm, params):
        raise NotImplementedException