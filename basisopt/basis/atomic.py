from mendeleev import element as md_element
from mendeleev.econf import ElectronicConfiguration
import functools
import numpy as np
import logging
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
    """Decorator that checks if the AtomicBasis has an element attribute
       Raises:
            ElementNotSet if no element found
    """
    @functools.wraps(func)
    def wrapper(basis, *args, **kwargs):
        if basis.element is None:
            raise ElementNotSet
        else:
            func(basis, *args, **kwargs)
    return wrapper

class AtomicBasis(Basis):
    """Object for preparation and optimization of a basis set for 
       a single atom. 
    
       Attributes:
            et_params (list): even tempered expansion parameters
            charge (int): net charge on atom
            multiplicity (int): spin multiplicity of atom
            config (dict): configuration of basis, (k, v) pairs
            of form (angular momentum: no. of functions)
            e.g. 's': 5, 'p': 4, etc.

       Special attribute:
            element-> gets Mendeleev Element object of atom (_element)
                   -> set with atomic symbol 
    
       Private Attributes:
            _element (mendeleev Element): object set via element
            _molecule (Molecule): Molecule object rep of atom
            _done_setup (bool): flag for whether ready for optimize
            _symbol (str): atomic symbol in lowercase
    """
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
            self._element = md_element(name.title())
            self._symbol = name.lower() 
            self._molecule.name = name + '_atom' 
            self._molecule._atom_names = [name]
            self._molecule._coords = [np.array([0.0, 0.0, 0.0])]
            self.results.name = self._molecule.name
        except:
            logging.error("Please enter a valid element")
            
    @property
    def charge(self):
        return self._charge

    @charge.setter
    @needs_element
    def charge(self, new_charge):
        nelec = self._element.electrons - new_charge
        if nelec < 1:
            logging.warning(f"A charge of {new_charge} would remove all electrons, setting to 0")
            self._charge = 0 
        else:
            self._charge = new_charge
        self._molecule.charge = self._charge

    @property
    def multiplicity(self):
        return self._multiplicity
    
    @needs_element
    def get_basis(self):
        return self._molecule.basis[self._symbol]

    @multiplicity.setter
    @needs_element
    def multiplicity(self, new_mult):
        if (new_mult < 1) or (new_mult-1 > self._element.electrons):
            logging.warning(f"Multiplicity can't be set to {new_mult}, setting to 1")
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
            logging.warning(f"Configuration {new_config} is insufficient, using minima config")
            self._config = minimal
        else:
            self._config = new_config
    
    def minimal(self):
        """Returns the minimal basis configuration for this atom"""
        if self._element is None:
            return {}
        else:
            return zt.minimal(self._element)
    
    @needs_element
    def configuration(self, quality='dz'):
        """Sets the basis set configuration to a desired quality
        
           Arguments:
                quality (str): name of quality type, see zetatools for options
        """
        try:
            zeta_func = zt.QUALITIES[quality.lower()]
            self._config = zeta_func(self._element)
            config_string = zt.config_to_string(self._config)
            logging.info(f"Primitive configuration of {config_string}")
        except KeyError:
            logging.warning(f"Could not find a {quality} strategy for configuration, using minimal")
            self._config = self.minimal()
    
    def as_xyz(self):
        """Returns the atom as an xyz file string"""
        return self._molecule.to_xyz()
    
    @needs_element
    def setup(self, method='ccsd(t)', quality='dz', strategy=Strategy(), reference=('cc-pvqz', None), params={}):
        """Sets up the basis ready for optimization. Must be called before optimize is called
        
           Arguments:
                method (str): the method to use; available methods can be checked via the Wrapper object
                quality (str): configuration quality, see zetatools for options
                strategy (Strategy): the optimization strategy to use, e.g. EvenTemperedStrategy
                reference (tuple): the reference value for the objective being calculated, either as
                (name, value) OR (basis_name, None). The latter will calculate the reference by downloading the
                requested basis from the BSE
                params (dict): dictionary of parameters to pass to the backend - see the relevant Wrapper object
                for options
        
            Sets:
                self.strategy
                self.config
                self._done_setup - cannot call optimize until this flag is True
        """
        # get configuration
        self.configuration(quality=quality)
        logging.info(f"Using the {strategy.name} building strategy")
        logging.info(f"Method: {method}")
        
        # Set or compute reference value
        label,value = reference
        self._molecule.method = method
        self.strategy = strategy
        logging.info(f"Reference type for this strategy is {strategy.eval_type}")
        if value is None:
            # Compute
            value = 0.0
            if api.which_backend() == 'Empty':
                logging.warning(f"No backend currently set, can't compute reference value")
            else:
                logging.info(f"Calculating reference value using {api.which_backend()} and {method}/{label}")
                self._molecule.basis  = fetch_basis(label, self._symbol)
                success = api.run_calculation(evaluate=strategy.eval_type, mol=self._molecule, params=params)
                if success != 0:
                    logging.warning("Reference calculation failed")
                else:
                    value = api.get_backend().get_value(strategy.eval_type)         
        self._molecule.add_reference(strategy.eval_type, value)
        logging.info(f"Reference value set to {value}")
        
        # Make a guess for the primitives
        logging.info(f"Generating starting guess from {strategy.guess.__name__}")
        self._molecule.basis[self._symbol] = strategy.guess(self, params=strategy.guess_params)
        self._done_setup = True
        logging.info("Setup complete")
        
    @needs_element
    def set_even_tempered(self, method='hf', accuracy=1e-5, max_n=18, max_l=-1, exact_ref=True, params={}):
        """Looks up or computes an even tempered basis expansion for the atom
        
           Arguments:
                method (str): method to use; possibilities can be found through Wrapper object
                accuracy (float): the tolerance to optimize to, compared to reference value
                max_n (int): max number of primitives per shell
                max_l (int): angular momentum to go up to; if -1, will use max l in minimal config
                exact_ref (bool): uses exact numerical HF energy if True, 
                calculates cc-pV5Z reference value if False 
                params (dict): dictionary of parameters to pass to the backend - 
                see the relevant Wrapper object for options
        
           Sets:
                self.et_params 
        """
        self.et_params = data.get_even_temper_params(atom=self._symbol.title(), accuracy=accuracy)
        if (len(self.et_params) == 0):
            # optimize new params
            if exact_ref:
                reference = ('exact', data._ATOMIC_HF_ENERGIES[self._element.atomic_number])
            else:
                reference = ('cc-pV5Z', None)
            strategy = EvenTemperedStrategy()
            self.setup(method=method, strategy=strategy, reference=reference, params=params)
            self.optimize(algorithm='Nelder-Mead', params=params)
            self.et_params = strategy.shells
        else:
            self._molecule.basis[self._symbol] = even_temper_expansion(self.et_params)
    
    @needs_element
    def optimize(self, algorithm='Nelder-Mead', params={}):
        """Runs the basis optimization
            
           Arguments:
                algorithm (str): optimization algorithm to use, see scipy.optimize for options
                params (dict): dictionary of parameters to pass to the backend - 
                see the relevant Wrapper object for options
        """
        if self._done_setup:
            optimize(self._molecule, algorithm=algorithm, strategy=self.strategy, **params)
        else:
            logging.error("Please call setup first")
        
    def contract(self):
        """Handles contraction of primitives"""
        raise NotImplementedException
    
        
