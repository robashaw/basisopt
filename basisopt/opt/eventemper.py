from basisopt import api, data
from basisopt.exceptions import PropertyNotAvailable
from basisopt.basis import even_temper_expansion
from basisopt.basis.guesses import null_guess
from .preconditioners import unit
from .strategies import Strategy
import numpy as np
from mendeleev import element as md_element

_INITIAL_GUESS = (0.3, 2.0, 8)

class EvenTemperedStrategy(Strategy):
    def __init__(self, eval_type='energy', target=1e-5, max_n=18, max_l=-1):
        Strategy.__init__(self, eval_type=eval_type, pre=unit)
        self.name = 'EvenTemper'
        self.shells = []
        self.target = target
        self.guess = null_guess
        self.guess_params = {}
        self.max_n = max_n
        self.max_l = max_l
        self.first_run = True
    
    def set_basis_shells(self, basis, element):
        basis[element] = even_temper_expansion(self.shells)
        
    def initialise(self, basis, element):
        if (self.max_l < 0):
            el = md_element(element.title())
        l_list = [l for (n, l) in el.ec.conf.keys()]
        min_l = len(set(l_list))
        
        self.max_l = max(min_l, self.max_l)
        self.shells = [_INITIAL_GUESS] * self.max_l
        self.shell_done = [1] * self.max_l
        self.set_basis_shells(basis, element)
        self.last_objective = 0
    
    def get_active(self, basis, element):
        (c, x, n) = self.shells[self._step]
        return np.array([c, x])
    
    def set_active(self, values, basis, element):
        (c, x, n) = self.shells[self._step]
        c = max(values[0], 1e-5)
        x = max(values[1], 1.01)
        self.shells[self._step] = (c, x, n)
        self.set_basis_shells(basis, element)
    
    def next(self, basis, element, objective):
        delta_objective = np.abs(self.last_objective - objective)
        self.last_objective = objective
        
        carry_on = True
        if self.first_run:
            self._step = self._step + 1
            if self._step == self.max_l:
                self.first_run = False
                self._step = 0
                (c, x, n) = self.shells[self._step]
                self.shells[self._step] = (c, x, min(n+1, self.max_n))
        else: 
            if (delta_objective < self.target):
                self.shell_done[self._step] = 0
            
            self._step = (self._step + 1) % self.max_l            
            (c, x, n) = self.shells[self._step]
            if (n == self.max_n):
                self.shell_done[self._step] = 0
            elif (self.shell_done[self._step] != 0):
                self.shells[self._step] = (c, x, n+1)
                
            carry_on = (np.sum(self.shell_done) != 0)
        
        return carry_on
            
                
    
