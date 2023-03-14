import numpy as np
import pytest

from basisopt.containers import *
from basisopt.exceptions import DataNotFound, InvalidResult
from tests.data import shells as shell_data
from tests.data.utils import almost_equal


def test_default_shell():
    new_shell = Shell()
    assert new_shell.l == 's'
    assert len(new_shell.coefs) == 0
    assert new_shell.exps.size == 0


def test_shell_compute():
    hbas = shell_data.get_vdz_internal()
    for ix, s in enumerate(hbas['h']):
        for c, v in shell_data._compute_values:
            value = s.compute(*c)
            assert almost_equal(value, v[ix], thresh=1e-10)


def test_basis_dict():
    hbas = shell_data.get_vdz_internal()
    d = basis_to_dict(hbas)
    assert 'h' in d
    b = dict_to_basis(d)
    assert 'h' in b
    for s, s_ in zip(hbas['h'], b['h']):
        assert s.exps.size == s_.exps.size


def test_default_result():
    r = Result()
    assert r.name == "Empty"
    assert r._depth == 1
    assert len(r._children) == 0


def test_add_get_data():
    r = Result(name='Test')
    assert r.name == 'Test'
    r.add_data("Is_Banana", True)
    assert r.get_data("Is_Banana")
    r.add_data("Is_Banana", False)
    assert not r.get_data("Is_Banana")
    assert r.get_data("Is_Banana", step_back=1)
    assert r.get_data("Is_Banana", step_back=4)

    with pytest.raises(DataNotFound):
        r.get_data("Is_Apple")


def build_frame():
    r1 = Result()
    r1.add_data("Is_Banana", True)
    r1.add_data("Is_Banana", False)
    r2 = Result(name='Child1')
    r2.add_data("Is_Banana", False)
    r2.add_data("Size", 10.1)
    r3 = Result(name='Child2')
    r3.add_data("Surname", "Flump")
    r4 = Result(name='Grandchild')
    r4.add_data("Size", 4.3)
    r4.add_data("Is_Banana", True)
    r1.add_child(r2)
    r1.add_child(r3)
    r2.add_child(r4)
    return r1, r2, r3, r4


def test_add_get_child():
    r1, r2, r3, r4 = build_frame()

    assert r1.depth == 1
    assert r2.depth == 2
    assert r3.depth == 2
    assert r4.depth == 3
    assert len(r1._children) == 2
    assert len(r2._children) == 1
    assert len(r3._children) == 0

    assert r1.get_child("Child1").depth == 2
    assert r1.get_child("Child2").depth == 2
    assert r2.get_child("Grandchild").depth == 3

    with pytest.raises(DataNotFound):
        r1.get_child("Grandchild")

    with pytest.raises(InvalidResult):
        shell = Shell()
        r3.add_child(shell)


def test_search_result():
    r1, r2, r3, r4 = build_frame()

    results = r1.search("Is_Banana")
    assert len(results) == 4
    assert not results['Child1_Is_Banana1']
    assert results['Grandchild_Is_Banana1']

    results = r3.search("Is_Banana")
    assert len(results) == 0

    results = r2.search("Size")
    assert len(results) == 2
    assert 4.3 in results.values()

    results = r1.search("Surname")
    assert "Flump" in results.values()

    results = r2.search("Surname")
    assert "Flump" not in results.values()


def test_load_result():
    r = Result().load("tests/data/result_test.bin")
    assert r.name == 'Parent'
    assert len(r._children) == 2
    assert r.get_data("age") == 32
    assert r.get_data("age", step_back=1) == 24
    assert len(r.get_data("position")) == 3
    assert r.get_data("height") == 150

    child1 = r.get_child("Child1")
    assert child1.get_data("name") == "Steven"
    assert child1.get_data("name", step_back=2) == "Sally"

    child2 = r.get_child("Child2")
    assert child2.get_data("age") == 2
    assert child2.depth == 2

    belongings = child2.get_child("Belongings")
    assert belongings.depth == 3
    assert belongings.get_data("toy")
