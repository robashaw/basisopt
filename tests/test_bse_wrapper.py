import basis_set_exchange as bse
import pytest

from basisopt import bse_wrapper as bsew
from tests.data import shells as shell_data


def test_make_bse_shell():
    internal_vdz = shell_data.get_vdz_internal()
    vdz_h = internal_vdz['h']

    bse_s_shell = bsew.make_bse_shell(vdz_h[0])
    assert bse_s_shell['angular_momentum'][0] == 0
    assert len(bse_s_shell['exponents']) == shell_data._nsexp
    assert len(bse_s_shell['coefficients']) == shell_data._nsfuncs


def test_make_internal_shell():
    bse_vdz = bse.get_basis('cc-pvdz', ['H'])['elements']['1']

    internal_s_shell = bsew.make_internal_shell(bse_vdz['electron_shells'][0])
    assert internal_s_shell.l == 's'
    assert len(internal_s_shell.exps) == shell_data._nsexp
    assert len(internal_s_shell.coefs) == shell_data._nsfuncs


def test_fetch_basis():
    internal_vdz_ref = shell_data.get_vdz_internal()
    internal_vdz_fetch = bsew.fetch_basis('cc-pvdz', ['H'])

    assert 'h' in internal_vdz_fetch.keys()
    h_ref = internal_vdz_ref['h']
    h_fetch = internal_vdz_fetch['h']

    assert len(h_ref) == len(h_fetch)
    for s1, s2 in zip(h_ref, h_fetch):
        assert shell_data.shells_are_equal(s1, s2)
