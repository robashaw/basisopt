import numpy as np

import basisopt.basis.guesses as guesses
from basisopt.basis.atomic import AtomicBasis
from tests.data.shells import get_vdz_internal, shells_are_equal
from tests.data.utils import almost_equal


def test_null_guess():
    o = AtomicBasis('O')
    result = guesses.null_guess(o)
    assert len(result) == 0


def logx_mean(shells):
    logx = []
    for s in shells:
        logxs = np.log(s.exps)
        logx = [*logx, *logxs]
    return np.mean(logx)


def test_log_normal_guess():
    # this has randomness so might fail?
    o = AtomicBasis().load('tests/data/oxygen-unopt.obj')

    # standard normal
    results = guesses.log_normal_guess(o)
    assert len(results) == 2
    assert len(results[0].exps) == 2
    assert len(results[1].exps) == 3
    assert np.abs(logx_mean(results)) <= 3.3  # 99.9 percentile

    # shifted normal
    results = guesses.log_normal_guess(o, params={'mean': 5.0, 'sigma': 0.1})
    assert len(results) == 2
    assert len(results[0].exps) == 2
    assert len(results[1].exps) == 3
    assert np.abs(logx_mean(results) - 5.0) <= 0.33


def test_bse_guess():
    h = AtomicBasis('H')
    results = guesses.bse_guess(h)
    vdz = get_vdz_internal()
    reference = vdz['h']
    for s1, s2 in zip(results, reference):
        assert shells_are_equal(s1, s2)


def test_even_temper_guess():
    ne = AtomicBasis().load('tests/data/neon-et.obj')
    results = guesses.even_tempered_guess(ne)
    assert len(results) == 2
    assert len(results[0].exps) == 18
    assert len(results[1].exps) == 13
    assert almost_equal(results[0].exps[0], 0.243032, thresh=1e-6)
    assert almost_equal(results[1].exps[0], 0.121994, thresh=1e-6)
