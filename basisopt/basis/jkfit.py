import numpy as np

from basisopt.util import bo_logger, dict_decode
from basisopt.bse_wrapper import fetch_basis
from basisopt.molecule import build_diatomic
from basisopt.opt import optimize, Strategy
from basisopt.opt.reduce import ReduceStrategy

from .basis import Basis

class JKFitBasis(Basis):
    """Object for preparation and optimization of a an auxiliary basis set
       for a single atom, for the fitting of the Coulomb (and optionally Exchange)
       integrals in a Hartree-Fock or DFT calculation. 
    """
    def __init__(self, name='H', charge=0, mult=1, mol=None, jonly=False):
        super(JKFitBasis, self).__init__()
        self.name = name
        if mol:
            self._molecule = mol
        else:
            mol_str = f"{name.title()}H,1.5"
            self._molecule = build_diatomic(mol_str, charge=charge, mult=mult)
        
        self.basis_type = "jkfit"
        if jonly:
            self.basis_type = "jfit"
        self._done_setup = False
    
    def as_dict(self):
        d = super(JKFitBasis, self).as_dict()
        d["@module"] = type(self).__module__
        d["@class"]  = type(self).__name__
        d["name"] = self.name
        d["basis_type"] = self.basis_type
        
        if hasattr(self, 'strategy'):
            if isinstance(self.strategy, Strategy):
                d["strategy"] = self.strategy
                d["done_setup"] = self._done_setup
        return d
    
    @classmethod
    def from_dict(cls, d):
        basis = Basis.from_dict(d)
        mol = basis._molecule
        charge = basis._molecule.charge
        mult = basis._molecule.multiplicity
        instance = cls(name=d.get("name", mol.name),
                       charge=charge, mult=mult, mol=mol)
        instance.results = basis.results
        instance.opt_results = basis.opt_results
        instance._tests = basis._tests
        instance._molecule = basis._molecule
        instance.basis_type = d.get("basis_type", "jkfit")
        instance.strategy = d.get("strategy", None)
        if instance.strategy:
            instance._done_setup = d.get("done_setup", False)
        return instance
    
    def get_basis(self):
        if self.basis_type == 'jfit':
            return self._molecule.jbasis
        return self._molecule.jkbasis
    
    def setup(self, basis, guess=None, config=None, method='rhf', params={}):
        """Sets up the basis ready for optimization. Must be called before optimize is called
        
           Arguments:
                basis
                guess
                config
                method
                params
        
           Sets:
                self.strategy
                self.config
                self._done_setup - cannot call optimize until this flag is True
        """
        if guess:
            if isinstance(guess, str):
                starting_basis = fetch_basis(guess, self._molecule.unique_atoms())
            else:
                starting_basis = guess
        else:
            bo_logger.error("No basis guess given")
            return
            
        if config:
            self.strategy = ReduceStrategy(starting_basis,
                                           eval_type='jk_error',
                                           method=method,
                                           target=1.,
                                           shell_mins=config, 
                                           max_l=-1,
                                           reopt_all=False,
                                           **params)
        else:
            self.strategy = Strategy(eval_type='jk_error')
        
        self.strategy.basis_type = self.basis_type
        self.strategy.orbital_basis = basis
        self._molecule.basis = basis
        self._molecule.method = method
        if self.basis_type == 'jfit':
            self._molecule.jbasis = starting_basis
        else:
            self._molecule.jkbasis = starting_basis
        self._done_setup = True
    
    def optimize(self, algorithm='Nelder-Mead', params={}):
        """Runs the basis optimization
            
           Arguments:
                algorithm (str): optimization algorithm to use, see scipy.optimize for options
                params (dict): dictionary of parameters to pass to the backend - 
                see the relevant Wrapper object for options
        """
        if self._done_setup:
            self.opt_results = optimize(self._molecule,
                                        element=self.name,
                                        algorithm=algorithm,
                                        strategy=self.strategy,
                                        **params)
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results
        

def jkfit_collection(element, starting_guess, basis_pairs=[],
                     charge=0, mult=1, mol=None, jonly=False,
                     method='rhf', algorithm='Nelder-Mead', 
                     opt_params={}, params={}):
    """ 
    """
    results = []
    guess = starting_guess
    for basis, config in basis_pairs:
        new_jk = JKFitBasis(name=element, charge=charge, mult=mult, mol=mol, jonly=jonly)
        new_jk.setup(basis, guess=guess, config=config, method=method, params=params)
        res = new_jk.optimize(algorithm=algorithm, params=opt_params)
        results.append(new_jk)
        if jonly:
            guess = new_jk._molecule.jbasis
        else:
            guess = new_jk._molecule.jkbasis
    return results