# containers

from . import data
from .exceptions import *
import pickle
import logging
import numpy as np
from scipy.special import sph_harm

class Shell:
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
        
    def compute(self, x, y, z, i=0, m=0):
        """ Computes the value of the (spherical) GTO at a given point
        
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
        r2 = x*x + y*y
        theta = np.arctan2(z, r2)
        r2 += z*z
        r = np.sqrt(r2)
        phi = np.arctan2(y, x)
        
        # Compute radial value
        radial_part = 0.0
        for x, c in zip(self.exps, self.coefs[i]):
            radial_part += c*np.exp(-x*r2)
        radial_part *= r**(lval)
        
        # Combine with angular value
        angular_part = np.real(sph_harm(m, lval, theta, phi))
        return (radial_part*angular_part)

class Result:
    """ Container for storing and archiving all results,
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
    def __init__(self, name='Empty'):
        self.name = name
        self._data_keys = {}
        self._data_values = {}
        self._children = []
        self.depth = 1
        
    def __str__(self):
        """Converts the Result into a human readable string
        
           Returns:
                a string representation of the Result object
        """
        string = f"{self.name} Results\n"
        
        # Print out all the immediate data
        ndat = len(self._data_keys.keys())
        string += f"\nDATA ({ndat} values)\n"
        for k, v in self._data_keys.items():
            value = self._data_values[k+str(v)]
            string += f"{k} = {value}\n"
            for n in range(v-1, 0, -1):
                value = self._data_values[k+str(n)]
                string += f"{k}@-{v-n} = {value}\n"
        
        # Recur over all children
        spacer = ["::"]*self._depth
        spacer = "".join(spacer)
        for child in self._children:
            string += "\n" + spacer + str(child)
        
        return string
        
    def statistics(self):
        """Tabulates summary statistics for the data in this Result
           Note: does not recur over children
        """
        raise NotImplementedException
    
    def _summary(self, title):
        """ Generates a summary string for the Result and all its children
        
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
    
    def summary(self):
        """Creates summaries of the Result and all its children
        
           Returns:
                a string with human-readable summary of the results
        """
        title_str = self.name + "->"
        return self._summary(title_str)
    
    @property
    def depth(self):
        return self._depth
        
    @depth.setter
    def depth(self, value):
        self._depth = value
        # Need to update all children too
        for c in self._children:
            c.depth = value + 1
    
    def add_data(self, name, value):
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
            self._data_values[name+"1"] = value
    
    def get_data(self, name, step_back=0):
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
        if name in self._data_keys:
            index = self._data_keys[name] - step_back
            index = max(1, index)
            return self._data_values[name+str(index)]
        else:
            # Have to raise an exception as we cannot surmise data type
            raise DataNotFound
    
    def add_child(self, child):
        """Adds a child Result to this Result"""
        if hasattr(child, '_depth'):
            child.depth = self.depth+1
            self._children.append(child)
        else:
            raise InvalidResult 
            
    def get_child(self, name):
        """Returns child Result with given name, if it exists"""
        for c in self._children:
            if c.name == name:
                return c
        raise DataNotFound
        
    def save(self, filename):
        """Pickles the Result object into a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
        logging.info(f"Dumped object of type {type(data)} to {filename}")
        
    def load(self, filename):    
        """Loads and returns a Result object from a file pickle"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            f.close()
        logging.info(f"Loaded object of type {type(data)} from {filename}")
        return data
