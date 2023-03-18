from typing import Any

import numpy as np
from monty.json import MSONable

from basisopt import api
from basisopt.basis.guesses import bse_guess
from basisopt.containers import InternalBasis, basis_to_dict
from basisopt.exceptions import PropertyNotAvailable
from basisopt.util import bo_logger, dict_decode

from .preconditioners import Preconditioner, make_positive


class Strategy(MSONable):
    """Object to describe and handle basis set optimization strategies.
    All strategy types should inherit from here, and give a description
    of the approach in the docs. This is a MINIMAL implementation, so all
    methods here should usually be overridden in child classes

    This class also acts as a 'Default' optimization strategy. The alg is
    as follows:

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: cc-pVDZ
        Pre-conditioner: any (default, make sure exponents are positive)

        No initialisation needed.
        Optimize each shell in increasing order of angular momentum,
        so self._step = l+1, ends when self._step = max_l+1
        No iteration by default.

    Attributes:
        name (str): identifier
        eval_type (str): property to evaluate
        params (dict): parameters for backend, see relevant Wrapper for options
        guess (func): function to generate starting guess exponents
        guess_params (dict): parameters to pass to guess
        pre (func): function to precondition exponents - must have an inverse attribute
        pre.params (dict): parameters to pass to the preconditioner
        last_objective (float): last value of objective function
        delta_objective (float): change in value of objective function from last step
        first_run (bool): if True, next is yet to be called
        basis_type (str): "orbital/jfit/jkfit", allows for strategy to be applied to auxiliary bases
        orbital_basis (dict): if using on auxiliary basis, need to specify orbital basis here

        loss (callable): function to calculate loss - currently fixed to RMSE

    Private attributes:
        _step (int): tracks what step of optimization we're on
    """

    def __init__(self, eval_type: str = 'energy', pre: Preconditioner = make_positive):
        self.name = 'Default'
        self._eval_type = ''
        self.eval_type = eval_type
        self.params = {}
        self.guess = bse_guess
        self.guess_params = {'name': 'cc-pvdz'}
        self._step = -1
        self.pre = pre
        self.pre.params = {}
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = True

        self.basis_type = 'orbital'
        self.orbital_basis = None

        # currently fixed, to be expanded later
        self.loss = np.linalg.norm

    @property
    def eval_type(self) -> str:
        return self._eval_type

    @eval_type.setter
    def eval_type(self, name: str):
        wrapper = api.get_backend()
        if name in wrapper.all_available():
            self._eval_type = name
        else:
            raise PropertyNotAvailable(name)

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """
        self._step = -1
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = True

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective
        self._step += 1
        self.first_run = False
        maxl = len(basis[element])
        return maxl != self._step

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of Strategy"""
        d = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "name": self.name,
            "eval_type": self._eval_type,
            "params": self.params,
            "guess_params": self.guess_params,
            "step": self._step,
            "pre_params": self.pre.params,
            "last_objective": self.last_objective,
            "delta_objective": self.delta_objective,
            "first_run": self.first_run,
            "basis_type": self.basis_type,
        }
        if self.orbital_basis:
            d['orbital_basis'] = basis_to_dict(self.orbital_basis)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates a Strategy from MSONable dictionary"""
        d = dict_decode(d)
        eval_type = d.get("eval_type", 'energy')
        instance = cls(eval_type=eval_type)
        instance.name = d.get("name", "Default")
        instance.params = d.get("params", {})
        instance.guess_params = d.get("guess_params", {})
        instance.pre_params = d.get("pre_params", {})
        instance._step = d.get("step", -1)
        instance.last_objective = d.get("last_objective", 0.0)
        instance.delta_objective = d.get("delta_objective", 0.0)
        instance.first_run = d.get("first_run", True)
        instance.basis_type = d.get("basis_type", "orbital")
        instance.orbital_basis = d.get("orbital_basis", None)
        bo_logger.warning(
            "Loading a Strategy from json uses default preconditioner and guess functions"
        )
        return instance
