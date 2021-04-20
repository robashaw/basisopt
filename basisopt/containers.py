# containers

from . import data, exceptions
import numpy as np
from scipy.special import sph_harm

class Shell:
    def __init__(self):
        self.l = 's'
        self.exps = []
        self.coefs = []
        
    def compute(self, x, y, z, m=0):
        r2 = x*x + y*y
        theta = np.arctan2(z, r2)
        r2 += z*z
        r = np.sqrt(r2)
        phi = np.arctan2(y, x)
        radial_part = 0.0
        for x, c in zip(self.exps, self.coefs):
            radial_part += c*np.exp(-x*r2)
        radial_part *= r**(self.l)
        angular_part = np.real(sph_harm(m, self.l, theta, phi))
        return radial_part*angular_part

class Result:
    def __init__(self, name='Empty'):
        self.name = name
        self._data_keys = {}
        self._data_values = {}
        self._children = []
        self.depth = 1
        
    def __str__(self):
        string = f"{self.name} Results\n"
        
        ndat = len(self._data_keys.keys())
        string += f"\nDATA ({ndat} values)\n"
        for k, v in self._data_keys.items():
            value = self._data_values[k+str(v)]
            string += f"{k} = {value}\n"
            for n in range(v-1, 0, -1):
                value = self._data_values[k+str(n)]
                string += f"{k}@-{v-n} = {value}\n"
        
        spacer = ["::"]*self._depth
        spacer = "".join(spacer)
        for child in self._children:
            string += "\n" + spacer + str(child)
        
        return string
        
    def statistics(self):
        return ""    
    
    def _summary(self, title):
        string = title.upper() + "\n"
        string += self.statistics()
        for c in self._children:
            child_title = title + c.name + "->"
            string += c._summary(child_title)
        return string
    
    def summary(self):
        title_str = self.name + "->"
        return self._summary(title_str)
    
    @property
    def depth(self):
        return self._depth
        
    @depth.setter
    def depth(self, value):
        self._depth = value
        for c in self._children:
            c.depth = value + 1
    
    def add_data(self, name, value):
        if name in self._data_keys:
            self._data_keys[name] += 1
            key = name + str(self._data_keys[name])
            self._data_values[key] = value
        else:
            self._data_keys[name] = 1
            self._data_values[name+"1"] = value
    
    def get_data(self, name, step_back=0):
        if name in self._data_keys:
            index = self._data_keys[name] - step_back
            index = max(1, index)
            return self._data_values[name+str(index)]
        else:
            raise DataNotFound
    
    def add_child(self, child):
        if hasattr(child, '_depth'):
            child.depth = self.depth+1
            self._children.append(child)
        else:
            raise InvalidResult 
            
    def get_child(self, name):
        for c in self._children:
            if c.name == name:
                return c
        raise DataNotFound
        
