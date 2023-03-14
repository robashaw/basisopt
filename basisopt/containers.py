# containers
import pickle
from typing import Any

import numpy as np
from monty.json import MSONable
from scipy.optimize import OptimizeResult
from scipy.special import sph_harm

from . import data
from .exceptions import DataNotFound, InvalidResult
from .util import bo_logger, dict_decode


class Shell(MSONable):
    """Lightweight container for basis set Shells.

    Attributes:
         l (char): the angular momentum name of the shell
         exps (numpy array, float): array of exponents
         coefs (list): list of numpy arrays of equal length to exps,
             corresponding to coefficients for each exponent
    """

    def __init__(self):
        self.l = 's'
        self.exps = np.array([])
        self.coefs = []

    def as_dict(self) -> dict[str, Any]:
        """Converts Shell to MSONable dictionary

        Returns:
            dictionary representation of Shell
        """
        d = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "l": self.l,
            "exps": self.exps,
            "coefs": self.coefs,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates Shell object from dictionary representation

        Arguments:
             d (dict): dictionary of Shell attributes

        Returns:
             Shell object
        """
        d = dict_decode(d)
        instance = cls()
        instance.l = d.get('l', 's')
        instance.exps = d.get('exps', np.array([]))
        instance.coefs = d.get('coefs', [])
        return instance

    def compute(self, x: float, y: float, z: float, i: int = 0, m: int = 0) -> float:
        """Computes the value of the (spherical) GTO at a given point

        Arguments:
            x, y, z (float): coordinates relative to center of GTO
            i (int): index of GTO in coefs
            m (int): azimuthal quantum number in [-l, l]

        Returns:
            The unnormalised value of the GTO at (x, y, z)
        """
        # bounds checking
        lval = data.AM_DICT[self.l]
        m = np.sign(m) * min(abs(m), lval)
        if i >= len(self.coefs):
            i = 0

        # Convert to spherical coords
        r2 = x * x + y * y
        theta = np.arctan2(z, r2)
        r2 += z * z
        r = np.sqrt(r2)
        phi = np.arctan2(y, x)

        # Compute radial value
        radial_part = 0.0
        for al, c in zip(self.exps, self.coefs[i]):
            radial_part += c * np.exp(-al * r2)
        radial_part *= r ** (lval)

        # Combine with angular value
        angular_part = np.real(sph_harm(m, lval, theta, phi))
        return radial_part * angular_part


InternalBasis = dict[str, list[Shell]]
BSEBasis = dict[str, Any]


def basis_to_dict(basis: InternalBasis) -> dict[str, Any]:
    """Converts an internal basis set of the form
    {atom: [shells]} to an MSONable dictionary

    Arguments:
         basis (dict): internal basis set

    Returns:
         json-writable dictionary
    """
    return {k: [s.as_dict() for s in v] for k, v in basis.items()}


def dict_to_basis(d: dict[str, Any]) -> InternalBasis:
    """Converts an MSON dictionary to an internal basis

    Arguments:
         d (dict): dictionary of basis set attributes

    Returns:
         internal basis set
    """
    if len(d) > 0:
        key = list(d.keys())[0]
        shell = d[key]
        if len(shell) > 0:
            obj = type(shell[0]).__name__
            if obj != 'Shell':
                return {k: [Shell.from_dict(s) for s in v] for k, v in d.items()}
    return d


OptResult = dict[str, OptimizeResult]
OptCollection = dict[str, OptResult]


class Result(MSONable):
    """Container for storing and archiving all results,
    e.g. of tests, calculations, and optimizations.

    Attributes:
        name (str): identifier for result
        depth (int): a Result object contains children,
            so a depth of 1 indicates no parents, 2 indicates
            one parent, etc.

    Private attributes:
        _data_keys (dict): dictionary with the format
            (value_name, number of records)
        _data_values (dict): dictionary of values with format
            (value_name_with_id, value)
        _children (list): references to child Result objects
    """

    def __init__(self, name: str = 'Empty'):
        self.name = name
        self._data_keys = {}
        self._data_values = {}
        self._children = []
        self.depth = 1

    def __str__(self) -> str:
        """Converts the Result into a human readable string

        Returns:
             a string representation of the Result object
        """
        string = f"{self.name} Results\n"

        # Print out all the immediate data
        ndat = len(self._data_keys.keys())
        string += f"\nDATA ({ndat} values)\n"
        for k, v in self._data_keys.items():
            value = self._data_values[k + str(v)]
            string += f"{k} = {value}\n"
            for n in range(v - 1, 0, -1):
                value = self._data_values[k + str(n)]
                string += f"{k}@-{v-n} = {value}\n"

        # Recur over all children
        spacer = ["::"] * self._depth
        spacer = "".join(spacer)
        for child in self._children:
            string += "\n" + spacer + str(child)

        return string

    def statistics(self):
        """Tabulates summary statistics for the data in this Result
        Note: does not recur over children
        """
        raise NotImplementedError

    def _summary(self, title: str) -> str:
        """Generates a summary string for the Result and all its children

        Arguments:
            title (str): title (usually name of object) to prepend to summary

        Returns:
            a summary string for the Result and its children
        """
        string = title.upper() + "\n"
        string += self.statistics()
        for c in self._children:
            child_title = title + c.name + "->"
            string += c._summary(child_title)
        return string

    def summary(self) -> str:
        """Creates summaries of the Result and all its children

        Returns:
             a string with human-readable summary of the results
        """
        title_str = self.name + "->"
        return self._summary(title_str)

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, value: int):
        self._depth = value
        # Need to update all children too
        for c in self._children:
            c.depth = value + 1

    def add_data(self, name: str, value: Any):
        """Adds a data point to the result, with archiving

        Arguments:
             name (str): identifier for the value
             value: the value, can be basically anything
        """
        if name in self._data_keys:
            # Archive previous results with same name
            self._data_keys[name] += 1
            key = name + str(self._data_keys[name])
            self._data_values[key] = value
        else:
            # Create an entry for this name
            self._data_keys[name] = 1
            self._data_values[name + "1"] = value

    def get_data(self, name: str, step_back: int = 0) -> Any:
        """Retrieve an archived data point

        Arguments:
             name (str): identifier for the value needed
             step_back(int): how many values back to go,
             default will return last point added (step_back=0)

        Returns:
             the value with the requested name, if it exists

        Raises:
             DataNotFound if the requested data doesn't exist
        """
        if name not in self._data_keys:
            # Have to raise an exception as we cannot surmise data type
            raise DataNotFound
        else:
            index = self._data_keys[name] - step_back
            index = max(1, index)
            return self._data_values[name + str(index)]

    def add_child(self, child: object):
        """Adds a child Result to this Result"""
        if hasattr(child, '_depth'):
            child.depth = self.depth + 1
            self._children.append(child)
        else:
            raise InvalidResult

    def get_child(self, name: str) -> object:
        """Returns child Result with given name, if it exists"""
        for c in self._children:
            if c.name == name:
                return c
        raise DataNotFound

    def search(self, name: str) -> dict[str, Any]:
        """Searches for all data in this and all its children
        with a given name, returning a dictionary indexed by
        the name and which child it was found in
        """
        results = {}
        if name in self._data_keys:
            for n in range(self._data_keys[name]):
                resname = f"{name}{n+1}"
                results[self.name + "_" + resname] = self._data_values[resname]
        for c in self._children:
            tmp = c.search(name)
            for k, v in tmp.items():
                results[k] = v
        return results

    def save(self, filename: str):
        """Pickles the Result object into a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
        bo_logger.info("Dumped object of type %s to %s", type(data), filename)

    def load(self, filename: str) -> object:
        """Loads and returns a Result object from a file pickle"""
        with open(filename, 'rb') as f:
            pkl_data = pickle.load(f)
            f.close()
        bo_logger.info("Loaded object of type %s from %s", type(pkl_data), filename)
        return pkl_data

    def as_dict(self) -> dict[str, Any]:
        """Converts Result (and all children) to an MSONable dictionary"""
        d = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "name": self.name,
            "data_keys": self._data_keys,
            "data_values": self._data_values,
            "depth": self.depth,
            "children": [],
        }

        for c in self._children:
            cd = c.as_dict()
            del cd["@module"]
            del cd["@class"]
            d['children'].append(cd)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates Result from dictionary representation,
        including recursive creation of children.
        """
        d = dict_decode(d)
        name = d.get("name", "Empty")
        instance = cls(name=name)
        instance._data_keys = d.get("data_keys", {})
        instance._data_values = d.get("data_values", {})
        instance.depth = d.get("depth", 1)
        children = d.get("children", [])
        for c in children:
            child = Result.from_dict(c)
            instance.add_child(child)
        return instance
