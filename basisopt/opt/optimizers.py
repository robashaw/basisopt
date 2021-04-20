from scipy.optimize import minimize
from .strategies import Strategy 
from basisopt import api
from basisopt.exceptions import FailedCalculation
import numpy as np


def _atomic_opt(basis, element, algorithm, strategy, opt_params, objective):
    print(f"Starting optimization of {element}/{strategy.eval_type}")
    print(f"Algorithm: {algorithm}, Strategy: {strategy.name}")
    objective_value = objective(strategy.get_active(basis, element))
    print(f"Initial objective value: {objective_value}")        
    while strategy.next(basis, element, objective_value):
        print(f"Doing step {strategy._step+1}")
        guess = strategy.get_active(basis, element)
        res = minimize(objective, guess, method=algorithm, **opt_params)
        objective_value = res.fun
        print(f"Parameters: {res.x}\nObjective: {objective_value}\n")
    return res

def optimize(molecule, element=None, algorithm='l-bfgs-b', strategy=Strategy(), reg=(lambda x: 0), opt_params={}):
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    def objective(x):
        strategy.set_active(x, molecule.basis, element)
        success = api.run_calculation(evaluate=strategy.eval_type, mol=molecule, params=strategy.params)
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        result = molecule.get_delta(strategy.eval_type)
        return np.linalg.norm(result) + reg(x)
    
    strategy.initialise(molecule.basis, element)
    return _atomic_opt(molecule.basis, element, algorithm, strategy, opt_params, objective)    
        
def collective_optimize(molecules, basis, opt_data=[], npass=3):
    wrapper = api.get_backend()
    results = []*len(opt_data)
    for i in range(npass):
        print(f"Collective pass {i+1}")
        total = 0.0
        for ix, (el, alg, strategy, reg, params) in enumerate(opt_data):
            def objective(x):
                strategy.set_active(x, basis, el)
                total = 0.0
                for mol in molecules:
                    mol.basis = basis
                    success = api.run_calculation(evaluate=strategy.eval_type, mol=mol, params=strategy.params)
                    if success != 0:
                        raise FailedCalculation
                    value = wrapper.get_value(strategy.eval_type)
                    name  = strategy.eval_type + "_" + el.title()
                    mol.add_result(name, value)
                    result = molecule.get_delta(name)
                    total += np.linalg.norm(result)
                return total + reg(x)
            
            if i == 0:
                strategy.initialise(basis, el)
            results[ix] = _atomic_opt(basis, el, alg, strategy, params, objective)
            total += results[ix].fun
        print(f'Collective objective: {total}')
    return results

    
            
                    
            