# zeta_tools
from basisopt import data
import numpy as np


QUALITIES = {}
def register_quality(func):
    QUALITIES[func.__name__] = func
    return func

def get_next_l(l_list):
    values = np.array([data.AM_DICT[l] for l in l_list])
    next_l = np.max(values) + 1
    return data.INV_AM_DICT[next_l]

def enum_shells(conf):
    l_list = [l for (n, l) in conf.keys()]
    l_list = set(l_list)
        
    config = dict((l, 0) for l in l_list)
    for (n, l) in conf.keys():
        config[l] += 1
    return config
    
def config_to_string(conf):
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
    config = enum_shells(el.ec.conf)
    valence = enum_shells(el.ec.get_valence().conf)
    for k, v in valence.items():
        config[k] += n*v
    return config

def add_np(config, n):
    for i in range(n):
        next_l = get_next_l(config.keys())
        config[next_l] = n-i
    return config

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
    
    
    
