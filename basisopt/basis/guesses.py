# ways of generating guesses for exponents
# NEEDS GREATLY EXPANDING
import numpy as np

from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import Shell

from .basis import even_temper_expansion, fix_ratio, uncontract_shell

# All guess functions need this signature
# func(atomic, params={}), where atomic is an AtomicBasis object
# and params is a dictionary of parameters. atomic must have attribute
# atomic.config set.
# Return an array of Shell objects (i.e. an internal basis for a single atom)


def null_guess(atomic, params={}):
    """Default guess type for testing, returns empty array"""
    return []


def log_normal_guess(atomic, params={'mean': 0.0, 'sigma': 1.0}):
    """Generates exponents randomly from a log-normal distribution

    Params:
         mean: centre of the log-normal distribution
         sigma: standard deviation of log-normal distribution
    """
    config = atomic.config
    basis = []
    for k, v in config.items():
        shell = Shell()
        shell.l = k
        shell.exps = np.random.lognormal(mean=params['mean'], sigma=params['sigma'], size=v)
        shell.exps = fix_ratio(shell.exps)
        uncontract_shell(shell)
        basis.append(shell)
    return basis


def bse_guess(atomic, params={'name': 'cc-pvdz'}):
    """Takes guess from an existing basis on the BSE

    Params:
         name (str): name of desired basis set
    """
    basis = fetch_basis(params['name'], [atomic._symbol])
    return basis[atomic._symbol]


def even_tempered_guess(atomic, params={}):
    """Takes guess from an even-tempered expansion

    Params:
         see signature for AtomicBasis.set_even_tempered
    """
    if atomic.et_params is None:
        atomic.set_even_tempered(**params)
    return even_temper_expansion(atomic.et_params)
