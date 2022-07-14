import numpy as np
from mendeleev import element as md_element

from basisopt import api, data
from basisopt.exceptions import PropertyNotAvailable
from basisopt.basis.basis import even_temper_expansion
from basisopt.basis.guesses import null_guess
from .preconditioners import unit
from .strategies import Strategy

_INITIAL_GUESS = (0.3, 2.0, 8)

class EvenTemperedStrategy(Strategy):
    """ Implements a strategy for an even tempered basis set, where each angular
        momentum shell is described by three parameters: (c, x, n)
        Each exponent in that shell is then given by
            y_k = c*(x**k) for k=0,...,n
        
        Algorithm: 
            Evaluate: energy (can change to any RMSE-compatible property)
            Loss: root-mean-square error
            Guess: null, uses _INITIAL_GUESS above
            Pre-conditioner: None
            
            Initialisation:
                - Find minimum no. of shells needed
                - max_l >= min_l
                - generate initial parameters for each shell
            
            First run:
                - optimize parameters for each shell once, sequentially
            
            Next shell in list not marked finished:
                - re-optimise
                - below threshold or n=max_n: mark finished
                - above threshold: increment n
            Repeat until all shells are marked finished. 
            
            Uses iteration, limited by two parameters:
                max_n: max number of exponents in shell
                target: threshold for objective function
    
        Additional attributes:
            shells (list): list of (c, x, n) parameter tuples
            shell_done (list): list of flags for whether shell is finished (0) or not (1)
            target (float): threshold for optimization delta
            max_n (int): maximum number of primitives in shell expansion
            max_l (int): maximum angular momentum shell to do;
            if -1, does minimal configuration     
    """
    def __init__(self, eval_type='energy', target=1e-5, max_n=18, max_l=-1):
        super(EvenTemperedStrategy, self).__init__(eval_type=eval_type, pre=unit)
        self.name = 'EvenTemper'
        self.shells = []
        self.shell_done = []
        self.target = target
        self.guess = null_guess
        self.guess_params = {}
        self.max_n = max_n
        self.max_l = max_l
        
    def as_dict(self):
        d = super(EvenTemperedStrategy, self).as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["shells"] = self.shells
        d["shell_done"] = self.shell_done
        d["target"] = self.target 
        d["max_n"] = self.max_n
        d["max_l"] = self.max_l
        return d
        
    @classmethod
    def from_dict(cls, d):
        strategy = Strategy.from_dict(d)
        instance = cls(
                       eval_type=d.get("eval_type", 'energy'),
                       target=d.get("target", 1e-5),
                       max_n=d.get("max_n", 18),
                       max_l=d.get("max_l", -1)
                      )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.shells = d.get("shells", [])
        instance.shell_done = d.get("shell_done", [])
        return instance
    
    def set_basis_shells(self, basis, element):
        """Expands parameters into a basis set"""
        basis[element] = even_temper_expansion(self.shells)
        
    def initialise(self, basis, element):
        if self.max_l < 0:
            el = md_element(element.title())
        l_list = [l for (n, l) in el.ec.conf.keys()]
        min_l = len(set(l_list))
        
        self.max_l = max(min_l, self.max_l)
        self.shells = [_INITIAL_GUESS] * self.max_l
        self.shell_done = [1] * self.max_l
        self.set_basis_shells(basis, element)
        self.last_objective = 0.
        self.delta_objective = 0.
        self.first_run = True
    
    def get_active(self, basis, element):
        (c, x, _) = self.shells[self._step]
        return np.array([c, x])
    
    def set_active(self, values, basis, element):
        (c, x, n) = self.shells[self._step]
        c = max(values[0], 1e-5)
        x = max(values[1], 1.01)
        self.shells[self._step] = (c, x, n)
        self.set_basis_shells(basis, element)
    
    def next(self, basis, element, objective):
        self.delta_objective = np.abs(self.last_objective - objective)
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
            if self.delta_objective < self.target:
                self.shell_done[self._step] = 0
            
            self._step = (self._step + 1) % self.max_l            
            (c, x, n) = self.shells[self._step]
            if n == self.max_n:
                self.shell_done[self._step] = 0
            elif self.shell_done[self._step] != 0:
                self.shells[self._step] = (c, x, n+1)
                
            carry_on = np.sum(self.shell_done) != 0
        
        return carry_on
            
                
    
