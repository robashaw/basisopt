# wrappers for BasisSetExchange functionality

import basis_set_exchange as bse
import basisopt.data as data
from basisopt.containers import Shell 
import numpy as np

def make_bse_shell(shell):
    """Converts an internal-format basis shell into a BSE-format shell
    
       Arguments:
            shell: an internal Shell object
    
       Returns:
            a BSE-format gto_spherical shell
    """
    new_shell = {
        "function_type" : 'gto_spherical',
        "region": "",
        "angular_momentum": [data.AM_DICT[shell.l]],
        "exponents": [f"{x:e}" for x in shell.exps],
        "coefficients": [[f"{c:e}" for c in arr] for arr in shell.coefs]
    }
    return new_shell

def make_internal_shell(shell):
    """Converts a BSE-format basis shell into an internal-format shell
    
       Arguments:
            shell: a BSE shell, a dictionary that must have these attributes
                   ['angular_momentum', 'exponents', 'coefficients']    
    
       Returns:
            an internal Shell object
    """
    new_shell = Shell()
    new_shell.l = data.INV_AM_DICT[shell['angular_momentum'][0]]
    new_shell.exps = np.array([float(x) for x in shell['exponents']])
    new_shell.coefs = [np.array([float(c) for c in arr]) for arr in shell['coefficients']]
    return new_shell

def internal_to_bse(basis):
    """Converts an internal basis dictionary into a BSE basis object
    
       Arguments:
            basis: an internal basis, which is a dictionary with k, v pairs like:
                   element_symbol: [array of internal Shell objects]
    
       Returns:
            a BSE basis of type 'component' with 'gto_spherical' function types
    """
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
    """Converts a BSE basis object into an internal basis dictionary
    
       Arguments:
            basis: a BSE basis, must have the following attributes
            ['elements'] each of which must then have an ['electron_shells']
            attribute
    
       Returns:
            an internal basis dictionary
    """
    new_basis = {}
    for z,e in basis['elements'].items():
        shells = [make_internal_shell(s) for s in e['electron_shells']]
        el = bse.lut.element_sym_from_Z(z)
        new_basis[el] = shells
    return new_basis

def internal_basis_converter(basis, fmt='gaussian94'):
    """Writes out an internal basis in the desired BSE format
    
       Arguments:
            basis (dict): the internal basis dictionary
            fmt (str): the desired output format - see the BSE docs for options
    
       Returns:
            the basis as a string in the desired format
    """
    bse_basis = internal_to_bse(basis)
    return bse.writers.write_formatted_basis_str(bse_basis, fmt)
    
def fetch_basis(name, elements):
    """Fetches a basis set for a set of elements from the BSE
    
       Arguments:
            name (str) - the name of the desired basis, see BSE docs for options
            elements (list) - a list of element symbols (or atomic numbers)
    
       Returns:
            an internal basis dictionary
    """
    basis = bse.get_basis(name, elements)
    return bse_to_internal(basis)