# funcitonality to rank basis shells
from basisopt import api
from basisopt.exceptions import FailedCalculation, EmptyBasis
from basisopt.basis import uncontract_shell
from basisopt.util import bo_logger
import numpy as np
import copy 
import logging

def rank_primitives(atomic, shells=None, eval_type='energy', params={}):
    """Systematically eliminates exponents from shells in an AtomicBasis
       to determine how much they contribute to the target property
    
       Arguments:
            atomic: AtomicBasis object
            shells (list): list of indices for shells in the AtomicBasis
            to be ranked. If None, will rank all shells
            eval_type (str): property to evaluate (e.g. energy)
            params (dict): parameters  to pass to the backend, 
            see relevant Wrapper for options
    
       Returns:
            (errors, ranks), where errors is a list of numpy arrays with the
            change in target property value for each exponent in the shell,
            and ranks is a list of numpy arrays which contain the indices of 
            each exponent in each shell from smallest to largest error value.
            Order of errors, ranks is same as order of shells
    
       Raises:
            FailedCalculation
    """
    mol = copy.copy(atomic._molecule)
    basis = mol.basis[atomic._symbol]
    if shells is None:
        shells = [s for s in range(len(basis))] # do all
    
    # Calculate reference value
    if(api.run_calculation(evaluate=eval_type, mol=mol, params=params) != 0):
        raise FailedCalculation
    else:
        reference = api.get_backend().get_value(eval_type)
        # prefix result  as being for ranking
        atomic._molecule.add_reference('rank_' + eval_type, reference)
    
    errors = []
    ranks  = []
    for s in shells:
        shell = basis[s]
        # copy old parameters
        exps  = shell.exps.copy()
        coefs = shell.coefs.copy()
        n = len(exps)
        
        # make uncontracted
        shell.exps = np.zeros(n-1)
        uncontract_shell(shell)
        err = np.zeros(n)
        
        # remove each exponent one at a time
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
        # reset shell to original
        shell.exps = exps
        shell.coefs = coefs
        
    return errors, ranks
    
def reduce_primitives(atomic, thresh=1e-4, shells=None, eval_type='energy', params={}):
    """Rank the primitive functions in an atomic basis, and remove those that contribute
       less than a threshold. TODO: add checking that does not go below minimal config
    
       Arguments:
            atomic: AtomicBasis object
            thresh (float): if a primitive's contribution to the target is < thresh,
            it is removed from the basis
            shells (list): list of indices of shells to be pruned; if None, does all shells
            eval_type (str): property to evaluate
            params (dict): parameters to pass to the backend
    
       Returns:
            (basis, delta) where basis is the pruned basis set (this is non-destructive to the
            original AtomicBasis), and delta is the change in target property with the pruned
            basis compared to the original
    
       Raises:
            FailedCalculation
    """
    mol = copy.copy(atomic._molecule)
    basis = mol.basis[atomic.symbol]
    if shells is None:
        shells = [s for s in range(len(basis))] # do all
    # first rank the primitives
    errors, ranks = rank_primitives(atomic, shells=shells, eval_type=eval_type, params=params)
    
    # now reduce
    for s, e, r in zip(shells, errors, ranks):
        shell = basis[s]
        n = shell.exps.size
        start = 0
        value = e[r[0]]
        while (start < n-1) and (value < thresh):
            start += 1
            loging.debug(f"{e}, {r}, {start}")
            value = e[r[start]]
        
        if start == (n-1):
            bo_logger.warning(f"Shell {s} with: l={shell.l}, x={shell.exps} now empty")
            shell.exps = []
            shell.coefs = []
        else:
            shell.exps = shell.exps[r[start:]]
            uncontract_shell(shell)
    
    success = api.run_calculation(evaluate=eval_type, mol=mol, params=params)
    result = api.get_backend().get_value(eval_type)
    delta = result - atomic._molecule.get_reference('rank_'+eval_type)
        
    return mol.basis, delta
