import logging
import numpy as np

from basisopt.testing.rank import *
from basisopt.basis.guesses import null_guess 
from basisopt.basis.basis import uncontract_shell
from basisopt.basis.atomic import AtomicBasis
from basisopt.util import bo_logger
from .strategies import Strategy
from .preconditioners import make_positive

class ReduceStrategy(Strategy):
    """TODO: Write strategy algorithm"""
    def __init__(self, starting_basis, eval_type='energy', method='scf', target=1e-5, shell_mins=[], max_l=-1, params={}):
        Strategy.__init__(self, eval_type=eval_type, pre=make_positive)
        self.name = 'Reduce'
        self.full_basis = starting_basis 
        self.shells = []
        self.target = target
        self.method = method
        self.guess  = null_guess
        self.guess_params = {}
        self.params = params
        self.shell_mins = shell_mins
        self.max_l = max_l
        self.nexps = []
        self.last_objective = 0
        self.reduction_step = True
        
    def set_basis_shells(self, basis, element):
        if element in self.full_basis:
            basis[element] = self.full_basis[element]
        else:
            basis[element] = {}
    
    def initialise(self, basis, element):
        self._step = -1
        self.set_basis_shells(basis, element)
        bel = basis[element]
        self.nexps = [len(s.exps) for s in bel]
        if self.max_l == -1:
            self.max_l = len(self.nexps)
        self.last_objective = 0
        self.reduction_step = True
        
    def next(self, basis, element, objective):
        carry_on = True
        if self.reduction_step:
            delta_objective = np.abs(self.last_objective - objective)
            self.last_objective = objective
            possible_changes = [(n - m) > 0 for n, m in zip(self.nexps, self.shell_mins)]
            
            carry_on = (delta_objective < self.target) and (True in possible_changes)
            if carry_on:
                at = AtomicBasis(name=element)
                at._molecule.basis = basis
                at._molecule.method = self.method
                errors, ranks = rank_primitives(at, eval_type=self.eval_type, params=self.params)
                min_errs = np.array([e[r[0]] for e, r in zip(errors, ranks)])
                for l in range(self.max_l):
                    if not possible_changes[l]:
                        min_errs[l] = np.max(min_errs)+1
                l = np.argmin(min_errs)
                ix = ranks[l][0]
                shell = basis[element][l]
                exps = shell.exps.copy()
                shell.exps = np.zeros(len(exps)-1)
                shell.exps[:ix] = exps[:ix]
                shell.exps[ix:] = exps[ix+1:] 
                uncontract_shell(shell)
                
                info_str = f"Removing exponent {exps[ix]} from shell with l={l}, error less than {min_errs[l]} Ha"
                bo_logger.info(info_str)
                self.reduction_step = False
        
        if carry_on:
            self._step += 1
            if self._step == self.max_l:
                self._step = -1
                self.reduction_step = True
        
        return carry_on