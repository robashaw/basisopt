# wrappers for BasisSetExchange functionality

import basis_set_exchange as bse
import basisopt.data as data
from basisopt.containers import Shell 
import numpy as np

def make_bse_shell(shell):
    new_shell = {
        "function_type" : 'gto_spherical',
        "region": "",
        "angular_momentum": [data.AM_DICT[shell.l]],
        "exponents": [f"{x:e}" for x in shell.exps],
        "coefficients": [[f"{c:e}" for c in arr] for arr in shell.coefs]
    }
    return new_shell

def make_internal_shell(shell):
    new_shell = Shell()
    new_shell.l = data.INV_AM_DICT[shell['angular_momentum'][0]]
    new_shell.exps = np.array([float(x) for x in shell['exponents']])
    new_shell.coefs = [np.array([float(c) for c in arr]) for arr in shell['coefficients']]
    return new_shell

def internal_to_bse(basis):
    # get a container
    bse_basis = bse.skel.create_skel('component')
    bse_basis['function_types'] = ['gto_spherical']
    elements = {}
    # add all the elements
    for el, b in basis.items():
        new_element = bse.skel.create_skel('element')
        new_element['electron_shells'] = [make_bse_shell(s) for s in b]
        z = bse.lut.element_Z_from_sym(el, as_str=True)
        elements[z] = new_element
        
    bse_basis['elements'] = elements
    return bse_basis
    
def bse_to_internal(basis):
    new_basis = {}
    for z,e in basis['elements'].items():
        shells = [make_internal_shell(s) for s in e['electron_shells']]
        el = bse.lut.element_sym_from_Z(z)
        new_basis[el] = shells
    return new_basis

def internal_basis_converter(basis, fmt='gaussian94'):
    bse_basis = internal_to_bse(basis)
    return bse.writers.write_formatted_basis_str(bse_basis, fmt)
    
def fetch_basis(name, elements):
    basis = bse.get_basis(name, elements)
    return bse_to_internal(basis)