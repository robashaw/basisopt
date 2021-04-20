from mendeleev import element as md_element
from mendeleev.econf import ElectronicConfiguration
import functools
import numpy as np
from . import zetatools as zt
from .basis import Basis, even_temper_expansion
from basisopt import api, data
from basisopt.molecule import Molecule
from basisopt.bse_wrapper import fetch_basis
from basisopt.opt.optimizers import optimize
from basisopt.opt.eventemper import EvenTemperedStrategy
from basisopt.opt.strategies import Strategy
from basisopt.exceptions import ElementNotSet


def needs_element(func):
    @functools.wraps(func)
    def wrapper(basis, *args, **kwargs):
        if basis.element is None:
            raise ElementNotSet
        else:
            func(basis, *args, **kwargs)
    return wrapper

class AtomicBasis(Basis):
    def __init__(self, name='H', charge=0, mult=1):
        Basis.__init__(self)

        self._element = None
        self._molecule = Molecule(name=name+'_atom')
        self.element = name
        self._done_setup = False
        self.et_params = None
        
        if self._element is not None:
            self.charge = charge
            self.multiplicity = mult
    
    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, name):
        try:
            self._element = md_element(name)
            self.symbol = name.lower() 
            self._molecule.name = name + '_atom' 
            self._molecule._atom_names = [name]
            self._molecule._coords = [np.array([0.0, 0.0, 0.0])]
            self.results.name = self._molecule.name
        except:
            print("Please enter a valid element")
            
    @property
    def charge(self):
        return self._charge

    @charge.setter
    @needs_element
    def charge(self, new_charge):
        nelec = self._element.electrons - new_charge
        if nelec < 1:
            print(f"A charge of {new_charge} would remove all electrons")
            self._charge = 0 
        else:
            self._charge = new_charge
        self._molecule.charge = self._charge

    @property
    def multiplicity(self):
        return self._multiplicity
    
    @needs_element
    def get_basis(self):
        return self._molecule.basis[self.symbol]

    @multiplicity.setter
    @needs_element
    def multiplicity(self, new_mult):
        if (new_mult < 1) or (new_mult-1 > self._element.electrons):
            print(f"Multiplicity can't be set to {new_mult}")
            self._multiplicity = 1
        else:
            self._multiplicity = new_mult
        self._molecule.multiplicity = self._multiplicity
            
    @property
    def config(self):
        return self._config        
    
    @config.setter
    def config(self, new_config):
        minimal = self.minimal()
        if zt.compare(minimal, new_config) < 0:
            print(f"Configuration {new_config} is insufficient")
            self._config = minimal
        else:
            self._config = new_config
    
    def minimal(self):
        if self._element is None:
            return {}
        else:
            return zt.minimal(self._element)
    
    @needs_element
    def configuration(self, quality='dz'):
        try:
            zeta_func = zt.QUALITIES[quality]
            self._config = zeta_func(self._element)
            config_string = zt.config_to_string(self._config)
            print(f"Primitive configuration of {config_string}")
        except KeyError:
            print(f"Could not find a {quality} strategy for configuration")
            self._config = None
    
    def as_xyz(self):
        return self._molecule.to_xyz()
    
    @needs_element
    def setup(self, method='ccsd(t)', quality='dz', strategy=Strategy(), reference=('cc-pvqz', None), params={}):
        # get configuration
        self.configuration(quality=quality)
        print(f"Using the {strategy.name} building strategy")
        print(f"Method: {method}")
        label,value = reference
        self._molecule.method = method
        self.strategy = strategy
        print(f"Reference type for this strategy is {strategy.eval_type}")
        if value is None:
            if api.which_backend() == 'Empty':
                print(f"No backend currently set, can't compute reference value")
            else:
                print(f"Calculating reference value using {api.which_backend()} and {method}/{label}")
                self._molecule.basis  = fetch_basis(label, self.symbol)
                success = api.run_calculation(evaluate=strategy.eval_type, mol=self._molecule, params=params)
                if success != 0:
                    print("Reference calculation failed")
                    value = 0.0
                else:
                    value = api.get_backend().get_value(strategy.eval_type)
                        
        self._molecule.add_reference(strategy.eval_type, value)
        print(f"Reference value set to {value}")
        
        print(f"Generating starting guess from {strategy.guess.__name__}")
        self._molecule.basis[self.symbol] = strategy.guess(self, params=strategy.guess_params)
        
        self._done_setup = True
        print("Setup complete")
        
    @needs_element
    def set_even_tempered(self, method='hf', accuracy=1e-5, max_n=18, max_l=-1, exact_ref=True):
        self.et_params = data.get_even_temper_params(atom=self.symbol.title(), accuracy=accuracy)
        if (len(self.et_params) == 0):
            # optimize new params
            if exact_ref:
                reference = ('exact', data._ATOMIC_HF_ENERGIES[self._element.atomic_number])
            else:
                reference = ('cc-pV5Z', None)
            strategy = EvenTemperedStrategy()
            self.setup(method=method, strategy=strategy, reference=reference)
            self.optimize(algorithm='Nelder-Mead')
            self.et_params = strategy.shells
        else:
            self._molecule.basis[self.symbol] = even_temper_expansion(self.et_params)
    
    @needs_element
    def optimize(self, algorithm='l-bfgs-b', params={}):
        if self._done_setup:
            optimize(self._molecule, algorithm=algorithm, strategy=self.strategy, **params)
        else:
            print("Please call setup first")
        
    def contract(self):
        raise NotImplementedException
    
        
