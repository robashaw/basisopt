import numpy as np
import pytest
from mendeleev import element

import basisopt.basis.zetatools as zt
from basisopt.basis.atomic import AtomicBasis

econf_1 = {(1, 's'): 2, (2, 's'): 2, (2, 'p'): 4}

econf_2 = {
    (1, 's'): 2,
    (2, 's'): 2,
    (2, 'p'): 6,
    (3, 's'): 2,
    (3, 'p'): 6,
    (4, 's'): 2,
    (3, 'd'): 10,
    (4, 'p'): 6,
    (5, 's'): 2,
    (4, 'd'): 10,
    (5, 'p'): 6,
    (4, 'f'): 14,
    (5, 'd'): 10,
    (6, 's'): 2,
}

conf_1 = {
    's': 4,
    'p': 3,
    'd': 2,
}

conf_2 = {'s': 5, 'p': 4, 'd': 3, 'f': 2, 'g': 1, 'h': 1}


def test_register_quality():
    assert len(zt.QUALITIES) > 0
    assert 'minimal' in zt.QUALITIES
    assert 'dzpp' in zt.QUALITIES


def test_get_next_l():
    ams = ['s']
    next_l = zt.get_next_l(ams)
    assert next_l == 'p'

    ams = ['s', 'p', 's', 'd', 'p']
    next_l = zt.get_next_l(ams)
    assert next_l == 'f'

    ams = ['g', 'f', 'd', 's']
    next_l = zt.get_next_l(ams)
    assert next_l == 'h'


def test_enum_shells():
    new_conf = zt.enum_shells(econf_1)
    assert new_conf['s'] == 2
    assert new_conf['p'] == 1

    new_conf = zt.enum_shells(econf_2)
    assert new_conf['s'] == 6
    assert new_conf['p'] == 4
    assert new_conf['d'] == 3
    assert new_conf['f'] == 1


def test_config_to_str():
    string1 = zt.config_to_string(conf_1)
    assert string1 == "4s3p2d"

    string2 = zt.config_to_string(conf_2)
    assert string2 == "5s4p3d2f1g1h"

    string3 = zt.config_to_string({})
    assert string3 == ""


def test_compare():
    assert zt.compare(conf_1, conf_2) > 0
    assert zt.compare(conf_1, conf_1) == 0
    assert zt.compare(conf_2, conf_1) < 0


def test_nz():
    o = element('O')
    conf = zt.nz(o, 2)
    assert conf['s'] == 3
    assert conf['p'] == 2
    assert 'd' not in conf


def test_add_np():
    new_conf = zt.add_np(conf_1, 2)
    assert 'f' in new_conf
    assert 'g' in new_conf
    assert 'h' not in new_conf

    new_conf = zt.add_np(conf_2, 1)
    assert 'i' in new_conf
