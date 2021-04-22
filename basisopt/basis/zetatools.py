# zeta_tools
from basisopt import data
import numpy as np

"""Dictionary of possible basis configuration 'qualities'
   A quality corresponds to a manner of calculating how many basis
   functions of each angular momentum there should be for an atom.
   E.g. a minimal quality has one function for each occupied atomic orbital

   Quality functions must have the signature
    func(Mendeleev Element) and return a configuration dictionary
    {'s': ns, 'p': np, etc.}

   List of qualities:
        minimal:            minimal basis configuration
        dz, tz, qz, n5z:    split valence
        dzp, tzp, qzp:      split valence polarized
        dzpp, tzpp, qzpp:   split valence double polarized
        cc_pvNz:            correlation consistent, N=d,t,q,5
"""
QUALITIES = {}

def register_quality(func):
    """Decorator to make a quality function available"""
    QUALITIES[func.__name__] = func
    return func

def get_next_l(l_list):
    """Given a list of existing angular momenta, gives
       the angular momentum symbol one higher.
    
       Arguments:
        l_list (list): (non-unique) list of angular momenta ['s', 'p', etc]
        
       Returns:
        angular momentum symbol one higher than max in l_list
    """
    values = np.array([data.AM_DICT[l] for l in l_list])
    next_l = np.max(values) + 1
    return data.INV_AM_DICT[next_l]

def enum_shells(conf):
    """Enumerates the number of functions of each angular momentum
    
       Arguments:
        conf: an ElectronConfig object from mendeleev
    
       Returns:
        a configuration dictionary 
    """
    l_list = [l for (n, l) in conf.keys()]
    l_list = set(l_list)
        
    config = dict((l, 0) for l in l_list)
    for (n, l) in conf.keys():
        config[l] += 1
    return config
    
def config_to_string(conf):
    """Converts a configuration dictionary into a string, e.g.
       '4s3p2d1f'
    """
    nl = len(conf.keys())
    values = [0]*nl
    for k, v in conf.items():
        values[data.AM_DICT[k]] = v
    value_string = ""
    for ix, v in enumerate(values):
        new_string = f"{v}{data.INV_AM_DICT[ix]}"
        value_string += new_string
    return value_string
    
def compare(c1, c2):
    """Compares two configuration dictionaries
    
       Returns:
            < 0 if c1 is bigger than c2
            0 if they're equivalent sizes
            > 0 if c2 is bigger than c1
    """
    result = len(c2.keys()) - len(c1.keys())
    while result > -1:
        for k, v in c1.items():
            delta = c2[k] - v
            if delta < 0:
                result = -1
            else:
                result += delta
        break
    return result

def nz(el, n):
    """Helper function to generate n-zeta split valence configs
    
       Arguments:
            el (Mendeleev element)
            n (int): split valence level, e.g. DZ=2, TZ=3, etc.
    
       Returns:
            a config dictionary
    """
    config = enum_shells(el.ec.conf)
    valence = enum_shells(el.ec.get_valence().conf)
    for k, v in valence.items():
        config[k] += n*v
    return config

def add_np(config, n):
    """Helper function to add n polarization functions
    
       Arguments:
            config: configuration dictionary to polarize
            n (int): no. of pol functions to add
    
       Returns:
            a configuration dictionary
    """
    for i in range(n):
        next_l = get_next_l(config.keys())
        config[next_l] = n-i
    return config

# Various quality functions are registered below
# please add name to list at head of file if you 
# add another

@register_quality
def minimal(el):
   return enum_shells(el.ec.conf)

@register_quality
def dz(el):
    return nz(el, 1)
     
@register_quality
def tz(el):
    return nz(el, 2)

@register_quality    
def qz(el):
    return nz(el, 3)
    
@register_quality
def n5z(el):
    return nz(el, 4)
    
@register_quality
def dzp(el):
    config = nz(el, 1)
    return add_np(config, 1)

@register_quality
def dzpp(el):
    config = nz(el, 1)
    return add_np(config, 2)
    
@register_quality
def tzp(el):
    config = nz(el, 2)
    return add_np(config, 1)

@register_quality
def tzpp(el):
    config = nz(el, 2)
    return add_np(config, 2)
    
@register_quality
def qzp(el):
    config = nz(el, 3)
    return add_np(config, 1)

@register_quality
def qzpp(el):
    config = nz(el, 3)
    return add_np(config, 2)
 
@register_quality  
def cc_pvdz(el):
    return dzp(el)

@register_quality
def cc_pvtz(el):
    return tzpp(el)

@register_quality
def cc_pvqz(el):
    config = qz(el)
    return add_np(config, 3)
    
@register_quality
def cc_pv5z(el):
    config = n5z(el)
    return add_np(config, 4)
    
    
    
