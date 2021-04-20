from basisopt.wrappers.wrapper import Wrapper
from basisopt.exceptions import MethodNotAvailable

_BACKENDS = dict()
_CURRENT_BACKEND = Wrapper()
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
        func()
    except KeyError:
        print(f"{name} is not a registered backend for basisopt")
        
def get_backend():
    return _CURRENT_BACKEND        
    
def set_tmp_dir(path, create=False):
    global _TMP_DIR
    _TMP_DIR = path
    
def get_tmp_dir():
    return _TMP_DIR

def which_backend():
    """Returns:
        str: The name of the currently registered backend
    """
    return _CURRENT_BACKEND._name

        
@register_backend
def psi4():
    """ Tests Psi4 import and prepares to be used as calculation backend"""
    print("Testing Psi4 import:")
    try:
        import psi4
        print("Success.")
        global _CURRENT_BACKEND
        from basisopt.wrappers.psi4 import Psi4Wrapper
        _CURRENT_BACKEND = Psi4Wrapper()
    except ImportError:
        print("Psi4 backend not found!")

@register_backend
def molpro():
    """ Tests molpro import and prepares to be used as calculation backend"""
    print("MOLPRO backend not currently implemented")

def run_calculation(evaluate='energy', mol=None, params={}):
    """ Interface to the wrapper used to run a calculation.
    
        Arguments:
            evaluate (str): The function to be called for the computation
            params (dict): A dictionary of parameters needed for the computation
    
        Returns:
            int: 0 on success, non-zero on failure
    """
    return _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)

