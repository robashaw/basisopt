from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import minimize

from basisopt import api
from basisopt.containers import InternalBasis, OptCollection, OptResult
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.util import bo_logger

from .regularisers import Regulariser
from .strategies import Strategy


def _atomic_opt(
    basis: InternalBasis,
    element: str,
    algorithm: str,
    strategy: Strategy,
    opt_params: dict[str, Any],
    objective: Callable[[np.ndarray], float],
) -> OptResult:
    """Helper function to run a strategy for a single atom

    Arguments:
         basis: internal basis dictionary
         element: symbol of atom to be optimized
         algorithm (str): optimization algorithm, see scipy.optimize for options
         opt_params (dict): parameters to pass to scipy.optimize.minimize
         objective (func): function to calculate objective, must have signature
             func(x) where x is a 1D numpy array of floats

     Returns:
         a dictionary of scipy.optimize result objects for each step in the opt
    """
    bo_logger.info("Starting optimization of %s/%s", element, strategy.eval_type)
    bo_logger.info("Algorithm: %s, Strategy: %s", algorithm, strategy.name)
    objective_value = objective(strategy.get_active(basis, element))
    bo_logger.info("Initial objective value: %f", objective_value)

    # Keep going until strategy says stop
    results = {}
    ctr = 1
    while strategy.next(basis, element, objective_value):
        bo_logger.info("Doing step %d", strategy._step + 1)
        guess = strategy.get_active(basis, element)
        if len(guess) > 0:
            res = minimize(objective, guess, method=algorithm, **opt_params)
            objective_value = res.fun
            info_str = "\n".join(
                [
                    f"Parameters: {res.x}",
                    f"Objective: {objective_value}",
                    f"Delta: {objective_value - strategy.last_objective}",
                ]
            )
            results[f"atomicopt{ctr}"] = res
            ctr += 1
        else:
            info_str = "Skipping empty shell"
        bo_logger.info(info_str)
    return results


def optimize(
    molecule: Molecule,
    element: Optional[str] = None,
    algorithm: str = 'l-bfgs-b',
    strategy: Strategy = Strategy(),
    reg: Regulariser = (lambda x: 0),
    opt_params: dict[str, Any] = {},
) -> OptResult:
    """General purpose optimizer for a single atomic basis

    Arguments:
        molecule: Molecule object
        element (str): symbol of atom to optimize; if None, will default to first atom in molecule
        algorithm (str): scipy.optimize algorithm to use
        strategy (Strategy): optimization strategy
        basis_type (str): which basis type to use; currently "orbital", "jfit", or "jkfit"
        reg (func): regularization function
        opt_params (dict): parameters to pass to scipy.optimize.minimize

    Returns:
        dictionary of scipy.optimize result objects for each step in the opt

    Raises:
        FailedCalculation
    """
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    basis = molecule.basis
    if strategy.basis_type == "jfit":
        basis = molecule.jbasis
    elif strategy.basis_type == "jkfit":
        basis = molecule.jkbasis

    def objective(x):
        """Set exponents, run calculation, compute objective
        Currently just RMSE, need to expand via Strategy
        """
        strategy.set_active(x, basis, element)
        success = api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        result = molecule.get_delta(strategy.eval_type)
        return strategy.loss(result) + reg(x)

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_opt(basis, element, algorithm, strategy, opt_params, objective)


OptData = tuple[str, str, Strategy, Regulariser, dict[str, Any]]


def collective_optimize(
    molecules: list[Molecule],
    basis: InternalBasis,
    opt_data: list[OptData] = [],
    npass: int = 3,
    parallel: bool = False,
) -> OptCollection:
    """General purpose optimizer for a collection of atomic bases

     Arguments:
          molecules (list): list of Molecule objects to be included in objective
          basis: internal basis dictionary, will be used for all molecules
          opt_data (list): list of tuples, with one tuple for each atomic basis to be
              optimized, (element, algorithm, strategy, regularizer, opt_params) - see the
              signature of _atomic_opt or optimize
          npass (int): number of passes to do, i.e. it will optimize each atomic basis
              listed in opt_data in order, then loop back and iterate npass times
          parallel (bool): if True, will try to run Molecule calcs in parallel

    Returns:
          dictionary of dictionaries of scipy.optimize results for each step,
          corresponding to tuple in opt_data

    Raises:
          FailedCalculation
    """
    results = {}
    for i in range(npass):
        bo_logger.info("Collective pass %d", i + 1)
        total = 0.0

        # loop over elements in opt_data, and collect objective into total
        ctr = 1
        for el, alg, strategy, reg, params in opt_data:

            def objective(x):
                """Set exponents, compute objective for every molecule in set
                Regularisation only applied once at end
                """
                strategy.set_active(x, basis, el)
                local_total = 0.0
                for mol in molecules:
                    mol.basis = basis

                results = api.run_all(
                    evaluate=strategy.eval_type,
                    mols=molecules,
                    params=strategy.params,
                    parallel=parallel,
                )
                for mol in molecules:
                    value = results[mol.name]
                    name = strategy.eval_type + "_" + el.title()
                    mol.add_result(name, value)
                    result = value - mol.get_reference(strategy.eval_type)
                    local_total += np.linalg.norm(result)
                return local_total + reg(x)

            strategy.initialise(basis, el)
            res = _atomic_opt(basis, el, alg, strategy, params, objective)
            total += strategy.last_objective
            results[f"pass{i}_opt{ctr}"] = res
            ctr += 1
        bo_logger.info('Collective objective: %f', total)
    return results
