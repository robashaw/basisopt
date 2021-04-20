# zeta_tools
from basisopt import data
from .basis import uncontract_shell, even_temper_expansion, fix_ratio
from basisopt.containers import Shell
from basisopt.bse_wrapper import fetch_basis 
import numpy as np

def null_guess(atomic, params={}):
    return []

def log_normal_guess(atomic, params={'mean' : 0.0, 'sigma': 1.0}):
    config = atomic.config
    basis = []
    for k, v in config.items():
        shell = Shell()
        shell.l = k
        shell.exps = np.random.lognormal(mean  = params['mean'], 
                                         sigma = params['sigma'],
                                         size  = v)
        shell.exps = fix_ratio(shell.exps)
        uncontract_shell(shell)
        basis.append(shell)
    return basis
    
def bse_guess(atomic, params={'name': 'cc-pvdz'}):
    basis = fetch_basis(params['name'], [atomic.symbol])
    return basis[atomic.symbol]

def even_tempered_guess(atomic, params={}):
    if atomic.et_params is None:
        atomic.set_even_tempered(**params)
    return even_temper_expansion(atomic.et_params)
    

        
        
    
    
    
