# base test types
from basisopt import api
from basisopt.exceptions import FailedCalculation, EmptyBasis
from basisopt.basis import uncontract_shell
import numpy as np
import copy 

def rank_primitives(atomic, shells=None, eval_type='energy', params={}):
    mol = copy.copy(atomic._molecule)
    basis = mol.basis[atomic.symbol]
    if shells is None:
        shells = [s for s in range(len(basis))] # do all
    
    if(api.run_calculation(evaluate=eval_type, mol=mol, params=params) != 0):
        raise FailedCalculation
    else:
        reference = api.get_backend().get_value(eval_type)
        atomic._molecule.add_reference('rank_' + eval_type, reference)
    
    errors = []
    ranks  = []
    for s in shells:
        shell = basis[s]
        exps  = shell.exps.copy()
        coefs = shell.coefs.copy()
        n = len(exps)
        shell.exps = np.zeros(n-1)
        uncontract_shell(shell)
        err = np.zeros(n)
        
        for i in range(n):
            shell.exps[:i] = exps[:i]
            shell.exps[i:] = exps[i+1:]
            success = api.run_calculation(evaluate=eval_type, mol=mol, params=params)
            if success != 0:
                raise FailedCalculation
            else:
                value = api.get_backend().get_value(eval_type)
                err[i] = np.abs(value - reference)
        
        errors.append(err)
        ranks.append(np.argsort(err))
        shell.exps = exps
        shell.coefs = coefs
        
    return errors, ranks
    
def reduce_primitives(atomic, thresh=1e-4, shells=None, eval_type='energy', params={}):
    mol = copy.copy(atomic._molecule)
    basis = mol.basis[atomic.symbol]
    if shells is None:
        shells = [s for s in range(len(basis))] # do all
    errors, ranks = rank_primitives(atomic, shells=shells, eval_type=eval_type, params=params)
    
    for s, e, r in zip(shells, errors, ranks):
        shell = basis[s]
        n = shell.exps.size
        start = 0
        value = e[r[0]]
        while (start < n-1) and (value < thresh):
            start += 1
            print(e, r, start)
            value = e[r[start]]
        
        if start == (n-1):
            print(f"Shell {s} with: l={shell.l}, x={shell.exps} now empty")
            shell.exps = []
            shell.coefs = []
        else:
            shell.exps = shell.exps[r[start:]]
            uncontract_shell(shell)
    
    success = api.run_calculation(evaluate=eval_type, mol=mol, params=params)
    result = api.get_backend().get_value(eval_type)
    delta = result - atomic._molecule.get_reference('rank_'+eval_type)
        
    return mol.basis, delta
