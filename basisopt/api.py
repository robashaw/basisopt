from basisopt.wrappers.wrapper import Wrapper
from basisopt.wrappers.dummy import DummyWrapper
from basisopt.exceptions import MethodNotAvailable
import logging
import colorlog
import os

try:
    import dask
    from basisopt.parallelise import *
    _PARALLEL = True
except:
    _PARALLEL = False

_BACKENDS = dict()
_CURRENT_BACKEND = DummyWrapper()
_TMP_DIR = ""


def register_backend(func):
    """Registers a function to set the backend for basisopt"""
    _BACKENDS[func.__name__] = func
    return func
    
def set_backend(name):
    """Sets the global backend for basisopt calculations
    
       Arguments:
            name (str): the name of the program to use 
    """
    try:
        func = _BACKENDS[name.lower()]
        if _CURRENT_BACKEND._name != "Dummy":
            logging.warning(f"Overwriting previous backend")
        func()
        logging.info(f"Backend set to {_CURRENT_BACKEND._name}")
    except KeyError:
        logging.error(f"{name} is not a registered backend for basisopt")
        
def get_backend():
    """Returns:
        backend: the Wrapper object for the current backend
    """
    return _CURRENT_BACKEND        
    
def set_tmp_dir(path):
    """Sets the working directory for all backend calculations,
       creating the directory if it doesn't already exist.
    
       Arguments:
            path (str): path to the scratch directory
    """
    global _TMP_DIR
    # check if dir exists, and create if not
    if not os.path.isdir(path):
        logging.info(f"Created directory at {path}")
        os.mkdir(path)
    _TMP_DIR = path
    logging.info(f"Scratch directory set to {_TMP_DIR}")
    
def get_tmp_dir():
    """Returns:
        Path to the current scratch/temp directory
    """
    return _TMP_DIR

def which_backend():
    """Returns:
        str: The name of the currently registered backend
    """
    return _CURRENT_BACKEND._name
    
def set_logger(level=logging.INFO, filename=None):
    """Initialises Python logging, formatting it nicely,
       and optionally printing to a file.
    """
    log_format = (
        '%(asctime)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format)
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

@register_backend
def dummy():
    """Sets backend to the DummyWrapper for testing and
       for when calculations aren't needed.
    """
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = DummyWrapper()

@register_backend
def psi4():
    """ Tests Psi4 import and prepares to be used as calculation backend"""
    try:
        global _CURRENT_BACKEND
        from basisopt.wrappers.psi4 import Psi4Wrapper
        _CURRENT_BACKEND = Psi4Wrapper()
    except ImportError:
        logging.error("Psi4 backend not found!")

@register_backend
def molpro():
    """ Tests molpro import and prepares to be used as calculation backend"""
    logging.error("MOLPRO backend not currently implemented")
    raise NotImplementedException

def run_calculation(evaluate='energy', mol=None, params={}):
    """ Interface to the wrapper used to run a calculation.
    
        Arguments:
            evaluate (str): The function to be called for the computation
            params (dict): A dictionary of parameters needed for the computation
    
        Returns:
            int: 0 on success, non-zero on failure
    """
    result = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
    _CURRENT_BACKEND.clean()
    return result

def _one_job(mol, evaluate='energy', params={}):
    success = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
    value   = (_CURRENT_BACKEND.get_value(evaluate))
    _CURRENT_BACKEND.clean()
    return mol.name,value

def run_all(evaluate='energy', mols=[], params={}, parallel=False):
    results = {}
    if parallel and _PARALLEL:
        kwargs = {"evaluate": evaluate, "params": params}
        with dask.config.set({"multiprocessing.context": "fork"}):
            tmp_results = distribute(3, _one_job, mols, **kwargs)
        for (n, v) in tmp_results:
            results[n] = v
    else:
        for m in mols:
            name, value = _one_job(m, evaluate=evaluate, params=params)
            results[name] = value
    return results
