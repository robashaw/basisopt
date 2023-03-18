import logging
import os

from basisopt import api
from basisopt.wrappers import Wrapper


def test_backend_registration():
    assert len(api._BACKENDS.keys()) > 0
    assert "dummy" in api._BACKENDS.keys()


def test_set_backend():
    api.set_backend("DuMmy")
    assert api.which_backend() == "Dummy"
    api.set_backend("NotABackendType")
    assert api.which_backend() == "Dummy"


def test_get_backend():
    assert isinstance(api.get_backend(), Wrapper)
    assert api.get_backend()._name == "Dummy"


def test_which_backend():
    assert api.which_backend() == "Dummy"


def test_get_set_tmp_dir():
    assert api.get_tmp_dir() == "."

    NEW_TMP = "_tmp/"
    api.set_tmp_dir(NEW_TMP)
    assert api.get_tmp_dir() == NEW_TMP
    assert os.path.isdir(NEW_TMP)

    api._TMP_DIR = ""
    api.set_tmp_dir(NEW_TMP)
    assert api.get_tmp_dir() == NEW_TMP

    if os.path.isdir(NEW_TMP):
        os.rmdir(NEW_TMP)


def test_set_logger():
    logger = logging.getLogger('basisopt')
    assert logger.getEffectiveLevel() == logging.INFO

    api.set_logger(level=logging.WARNING)
    assert logger.getEffectiveLevel() == logging.WARNING
