import logging
import numpy as np
import copy

from basisopt.testing.rank import *
from basisopt.basis.guesses import null_guess 
from basisopt.basis.basis import uncontract_shell
from basisopt.basis.atomic import AtomicBasis
from basisopt.util import bo_logger
from basisopt.containers import basis_to_dict, dict_to_basis
from .strategies import Strategy
from .preconditioners import make_positive

class ReduceStrategy(Strategy):
    """ Strategy that takes a basis set and systematically removes least important exponents,
        until either the change in objective is larger than a threshold value, or a minimal
        number of exponents is reached.
        
        Algorithm:
            Evaluate: energy (can change to any RMSE-compatible property)
            Loss: root-mean-square error
            Guess: none - initial basis set to reduce must be given
            Pre-conditioner: any (default, make sure exponents are positive)
            
            Initialization required, to determined parameters for reduction. 
            While delta_objective is below threshold, and basis size > minimal:
                - rank exponents by contribution to objective, for each shell 
                  that isn't already at its minimum size
                - remove the least important exponent, adjust basis size
                - reoptimize each shell in ascending angular-momentum order
                - recalculate delta_objective
            
            If delta_objective > threshold:
                - reset to basis set from previous step
    
        Attributes:
            full_basis (dict): internal basis to be reduced
            saved_basis (dict): internal basis from last step
            shells (list(int)): list of shells to be reduced
            target (float): maximum allowed change in objective value
            method (str): method used to evaluate objective
            shell_mins (list(int)): minimum number of exponents in each shell,
                    in ascending angular momentum order
            max_l (int): maximum angular momentum (inclusive) to reduce
            nexps (list(int)): number of exponents in each ang. momentum shell
            reduction_step (bool): if True, an exponent will be removed when next is called
    """
    def __init__(self, starting_basis, eval_type='energy', method='scf', target=1e-5, shell_mins=[], max_l=-1, params={}):
        Strategy.__init__(self, eval_type=eval_type, pre=make_positive)
        self.name = 'Reduce'
        self.full_basis = starting_basis 
        self.saved_basis = None
        self.shells = []
        self.target = target
        self.method = method
        self.guess  = self._guess
        self.guess_params = {}
        self.params = params
        self.shell_mins = shell_mins
        self.max_l = max_l
        self.nexps = []
        self.reduction_step = True
        
    def _guess(self, atomic, params={}):
        """Internal 'guess' returning the original unreduced basis"""
        return self.full_basis[atomic._symbol]
            
    def as_dict(self):
        d = super(ReduceStrategy, self).as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["full_basis"] = basis_to_dict(self.full_basis)
        d["saved_basis"] = basis_to_dict(self.saved_basis)
        d["target"] = self.target 
        d["max_l"] = self.max_l
        d["shells"] = self.shells
        d["method"] = self.method
        d["shell_mins"] = self.shell_mins
        d["nexps"] = self.nexps
        d["reduction_step"] = self.reduction_step
        return d
        
    @classmethod
    def from_dict(cls, d):
        strategy = Strategy.from_dict(d)
        full_basis = dict_to_basis(d.get("full_basis", {}))
        saved_basis = dict_to_basis(d.get("saved_basis", {}))
        instance = cls(
                       full_basis,
                       eval_type=d.get("eval_type", 'energy'),
                       method=d.get("method", 'scf'),
                       target=d.get("target", 1e-5),
                       shell_mins=d.get("shell_mins", []),
                       max_l=d.get("max_l", -1),
                       params=strategy.params
                      )
        instance.saved_basis = saved_basis
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.reduction_step = d.get("reduction_step", True)
        instance.shells = d.get("shells", [])
        return instance
        
    def set_basis_shells(self, basis, element):
        if element in self.full_basis:
            basis[element] = self.full_basis[element]
        else:
            basis[element] = {}
    
    def initialise(self, basis, element):
        self._step = -1
        self.first_run = True
        self.set_basis_shells(basis, element)
        bel = basis[element]
        self.nexps = [len(s.exps) for s in bel]
        if self.max_l == -1:
            self.max_l = len(self.nexps)-1
        self.last_objective = 0.
        self.delta_objective = 0.
        self.reduction_step = True
        
    def next(self, basis, element, objective):
        carry_on = True
        if self._step == self.max_l:
            self._step = -1
            self.reduction_step = True
        
        if self.reduction_step:
            if self.first_run:
                self.last_objective = objective
                self.first_run = False
                
            self.saved_basis = copy.deepcopy(self.full_basis)
            self.delta_objective = np.abs(self.last_objective - objective)
            self.last_objective = objective
            possible_changes = [(n - m) > 0 
                                for n, m in zip(self.nexps, self.shell_mins)]
            
            carry_on = (self.delta_objective < self.target)\
                             and (True in possible_changes)
            if carry_on:
                at = AtomicBasis(name=element)
                at._molecule.basis = basis
                at._molecule.method = self.method
                shells_to_rank = [s for s in range(self.max_l+1)
                                  if (basis[element][s].exps.size != 0)
                                  and possible_changes[s]]
                errors, ranks = rank_primitives(at, shells=shells_to_rank,
                                eval_type=self.eval_type, params=self.params)
                min_errs = np.array([e[r[0]] for e, r in zip(errors, ranks)])
                min_ix = np.argmin(min_errs)
                l = shells_to_rank[min_ix]
                ix = ranks[min_ix][0]
                shell = basis[element][l]
                exps = shell.exps.copy()
                shell.exps = np.zeros(len(exps)-1)
                shell.exps[:ix] = exps[:ix]
                shell.exps[ix:] = exps[ix+1:] 
                uncontract_shell(shell)
                
                info_str = f"Removing exponent {exps[ix]} from shell with l={l}, error less than {min_errs[l]} Ha"
                self.nexps[l] -= 1
                bo_logger.info(info_str)
                self.reduction_step = False
        
        if carry_on:
            self._step += 1
        else:
            if self.delta_objective > self.target:
                bo_logger.info("Change in objective over target, reverting to basis from last step")
                basis[element] = self.saved_basis[element]
            else:
                bo_logger.info("Reached minimum basis size")
            bo_logger.info("Finished reduction")
        
        return carry_on