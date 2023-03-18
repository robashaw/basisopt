# Template for program wrappers
import functools
from typing import Any, Callable

from basisopt.exceptions import InvalidMethodString, MethodNotAvailable
from basisopt.molecule import Molecule
from basisopt.util import bo_logger

Method = Callable[[object, Molecule, str, ...], Any]


def available(func: Method) -> Method:
    """Decorator to mark a method as available"""
    func._available = True
    return func


def unavailable(func: Method) -> Method:
    """Decorator to mark a method as unavailable"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise MethodNotAvailable(func.__name__)

    wrapper._available = False
    return wrapper


class Wrapper:
    """Abstract class to derive all backend wrappers from;
    see e.g. Psi4.

    All new calculation types must be added to this class and marked as unavailable,
    this means that the library knows what calculation types are possible
    agnostic to which wrapper is being used.

    You only have to implement the calculation types that you want to expose, and
    decorate them as being available. These functions should have the signature
     func(self, mol, tmp="") where mol is a Molecule object, and tmp is the path to
    the scratch directory.

    Attributes that should be set in children:
        _name (str): identifier, e.g. 'Psi4'
        _method_strings (dict): dictionary of method names and what calculation types
        can be done with them, e.g. {'hf': ['energy', 'dipole'], 'mp2': ['energy'], ...}

    Attributes used by children:
        _values (dict): dictionary where most recent calculated results are stored
        _globals (dict): dictionary of parameters that should be set every time
        a calculation is run, e.g. {'memory': '2gb', ...}. These should be parsed
        as part of the 'run' function in every Child implementation

    Attributes that should only be set here:
        _methods (dict): dictionary of all possible calculation types, pointing to member funcs
    """

    def __init__(self, name: str = 'Empty'):
        self._name = name

        self._methods = {
            'energy': self.energy,
            'dipole': self.dipole,
            'quadrupole': self.quadrupole,
            'trans_dipole': self.trans_dipole,
            'trans_quadrupole': self.trans_quadrupole,
            'polarizability': self.polarizability,
            'jk_error': self.jk_error,
        }

        self._method_strings = {}

        self._values = {}
        self._globals = {}

    def add_global(self, name: str, value: Any):
        """Add a global option"""
        self._globals[name] = value

    def get_value(self, name: str) -> Any:
        """Retrieve a data point if it exists"""
        if name in self._values:
            return self._values[name]
        return None

    def verify_method_string(self, string: str) -> bool:
        """Checks whether a method is available with this wrapper

        Arguments:
             string (str): a string of the form "name.method", e.g. "rhf.energy"
             will check to see if 'energy' can be calculated with 'rhf'

        Returns:
             True if available, False otherwise

        Raises:
             InvalidMethodString
        """
        parts = string.split('.')
        if len(parts) < 2:
            raise InvalidMethodString

        name = parts[0]
        method = parts[1]
        available = name in self._method_strings
        if available:
            methods = self._method_strings[name]
            available = method in methods
        return available

    def clean(self):
        """Cleans up any temporary files"""
        pass

    def run(self, evaluate: str, molecule: Molecule, params: dict[str, Any], tmp: str = "") -> int:
        """Runs a calculation with this backend
        MUST BE IMPLEMENTED IN ALL WRAPPERS

        Arguments:
             evaluate (str): the property to evaluate, e.g. 'energy'
             molecule: a Molecule object to run a calculation on
             params (dict): any parameters for the calculation in addition to _globals
             tmp (str): path to scratch directory

        Returns:
             0 on success, -1 if method isn't available, -2 otherwise
        """
        method_str = f"{molecule.method}.{evaluate}".lower()
        try:
            if self.verify_method_string(method_str):
                self._values[evaluate] = self._methods[evaluate](molecule, tmp=tmp, **params)
                return 0
            raise MethodNotAvailable(method_str)
        except KeyError as e:
            bo_logger.error(e)
            return -2
        except MethodNotAvailable:
            bo_logger.error("Unable to run %s with %s backend", method_str, self._name)
            return -1

    def method_is_available(self, method: str = 'energy') -> bool:
        """Returns True if a calculation type is available, false otherwise"""
        try:
            func = self._methods[method]
            return func._available
        except KeyError:
            return False

    def all_available(self) -> list[str]:
        """Returns a list of all available calculation types"""
        return [k for k, v in self._methods.items() if v._available]

    def available_properties(self, name: str) -> list[str]:
        """Returns a list of all available calculation types for a
        given method.

        Attributes:
             name (str): method name, e.g. 'rhf', 'mp2'
        """
        if name in self._method_strings:
            return self._method_strings[name]
        return []

    def available_methods(self, prop: str) -> list[str]:
        """Returns a list of all available methods to calculate a particular property

        Attributes:
             prop (str): name of property, e.g. 'energy', 'dipole'
        """
        return [k for k, v in self._method_strings.items() if prop in v]

    @unavailable
    def energy(self, mol, tmp="", **params):
        """Energy, Hartree"""
        raise NotImplementedError

    @unavailable
    def dipole(self, mol, tmp="", **params):
        """Dipole moment, numpy array, a.u."""
        raise NotImplementedError

    @unavailable
    def quadrupole(self, mol, tmp="", **params):
        """Quadrupole moment, numpy array, a.u."""
        raise NotImplementedError

    @unavailable
    def trans_dipole(self, mol, tmp="", **params):
        """Transition dipole moment, numpy array, a.u."""
        raise NotImplementedError

    @unavailable
    def trans_quadrupole(self, mol, tmp="", **params):
        """Transition quadrupole moment, numpy array, a.u."""
        raise NotImplementedError

    @unavailable
    def polarizability(self, mol, tmp="", **params):
        """Dipole polarizability, a.u."""
        raise NotImplementedError

    @unavailable
    def jk_error(self, mol, tmp="", **params):
        "JK density fitting error, Hartree"
        raise NotImplementedError
