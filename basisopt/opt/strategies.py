from basisopt import api, data
from basisopt.exceptions import PropertyNotAvailable
from .preconditioners import make_positive
from basisopt.basis.guesses import bse_guess
import numpy as np

class Strategy:
    def __init__(self, eval_type='energy', pre=make_positive):
        self.name = 'Default'
        self._eval_type = ''
        self.eval_type = eval_type
        self.params = {}
        self.guess = bse_guess
        self.guess_params = {'name': 'cc-pvdz'}
        self._step = -1
        self.pre = pre
        self.pre.params = {}
    
    @property
    def eval_type(self):
        return self._eval_type
    
    def initialise(self, basis, element):
        pass
    
    @eval_type.setter
    def eval_type(self, name):
        wrapper = api.get_backend()
        if name in wrapper.all_available:
            self._eval_type = name
        else:
            raise PropertyNotAvailable(name)
    
    def get_active(self, basis, element):
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)
        
    def set_active(self, values, basis, element):
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)
        
    def next(self, basis, element, objective):
        self._step += 1
        maxl = len(basis[element])-1
        return (maxl != self._step) 
            
    