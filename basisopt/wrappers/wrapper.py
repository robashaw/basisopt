# Template for program wrappers
from basisopt.exceptions import MethodNotAvailable
import functools

def available(func):
    func._available = True
    return func

def unavailable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise MethodNotAvailable(func.__name__)
    wrapper._available = False
    return wrapper

class Wrapper:
    def __init__(self, name='Empty'):
        self._name = name
        
        self._methods = {
            'energy'           : self.energy,
            'dipole'           : self.dipole,
            'quadrupole'       : self.quadrupole,
            'trans_dipole'     : self.trans_dipole,
            'trans_quadrupole' : self.trans_quadrupole,
            'polarizability'   : self.polarizability,
        } 
        
        self._method_strings = {}
        
        self._values = {}
        self._globals = {}
    
    def add_global(self, name, value):
        self._globals[name] = value
    
    def get_value(self, name):
        if name in self._values:
            return self._values[name]
        else:
            return None
    
    def verify_method_string(self, str):
        parts = str.split('.')
        name = parts[0]
        method = parts[1]
        available = (name in self._method_strings)
        if available:
            methods = self._method_strings[name]
            available = (method in methods)
        return available
    
    def run(self, evaluate, molecule, params, tmp=""):
        method_str = f"{molecule.method}.{evaluate}".lower()
        try:
            if self.verify_method_string(method_str): 
                self._values[evaluate] = self._methods[evaluate](mol=molecule, tmp=tmp, **params)
                return 0
            else:
                raise MethodNotAvailable(method_str)
        except KeyError:
            print(f"There is no method {evaluate}")
        except MethodNotAvailable:
            print(f"Unable to run {method_str} with {self._name} backend")
            return -1
    
    def method_is_available(self, method='energy'):
        try:
            func = self._methods[method]
            return func._available
        except KeyError:
            return False
    
    @functools.cached_property        
    def all_available(self):
        return [k for k, v in self._methods.items() if v._available]
        
    def available_properties(self, name):
        if name in self._method_strings:
            return self._method_strings[name]
        else:
            return []
    
    @functools.cache
    def available_methods(self, prop):
        return [k for k, v in self._method_strings.items() if prop in v]
        
    @unavailable
    def energy(mol=None, tmp=""):
        raise NotImplementedException
        
    @unavailable
    def dipole(mol=None, tmp=""):
        raise NotImplementedException
        
    @unavailable
    def quadrupole(mol=None, tmp=""):
        raise NotImplementedException
        
    @unavailable
    def trans_dipole(mol=None, tmp=""):
        raise NotImplementedException
    
    @unavailable
    def trans_quadrupole(mol=None, tmp=""):
        raise NotImplementedException
    
    @unavailable
    def polarizability(mol=None, tmp=""):
        raise NotImplementedException
    
    