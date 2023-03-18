import functools
import pickle
from typing import Any, Callable, Optional

import numpy as np
from mendeleev import element as MDElement

from basisopt import api, data
from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import InternalBasis, OptResult
from basisopt.exceptions import ElementNotSet
from basisopt.molecule import Molecule
from basisopt.opt.eventemper import EvenTemperedStrategy
from basisopt.opt.optimizers import optimize
from basisopt.opt.strategies import Strategy
from basisopt.util import bo_logger

from . import zetatools as zt
from .basis import Basis, even_temper_expansion


def needs_element(func: Callable) -> Callable:
    """Decorator that checks if the AtomicBasis has an element attribute
    Raises:
         ElementNotSet if no element found
    """

    @functools.wraps(func)
    def wrapper(basis, *args, **kwargs):
        if basis.element is None:
            raise ElementNotSet
        func(basis, *args, **kwargs)

    return wrapper


class AtomicBasis(Basis):
    """Object for preparation and optimization of a basis set for
    a single atom.

    Attributes:
         et_params (data.ETParams): even tempered expansion parameters
         charge (int): net charge on atom
         multiplicity (int): spin multiplicity of atom
         config (dict): configuration of basis, (k, v) pairs
             of form (angular momentum: no. of functions)
             e.g. 's': 5, 'p': 4, etc.

    Special attribute:
         element: gets Mendeleev Element object of atom (_element)
                  set with atomic symbol

    Private Attributes:
         _element (mendeleev Element): object set via element
         _molecule (Molecule): Molecule object rep of atom
         _done_setup (bool): flag for whether ready for optimize
         _symbol (str): atomic symbol in lowercase
    """

    def __init__(self, name: str = 'H', charge: int = 0, mult: int = 1):
        super().__init__()

        self._element = None
        self._molecule = Molecule(name=name + '_atom')
        self.element = name
        self._done_setup = False
        self.et_params = None

        if self._element is not None:
            self.charge = charge
            self.multiplicity = mult

    def save(self, filename: str):
        """Pickles the AtomicBasis object into a binary file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
        bo_logger.info("Dumped object of type %s to %s", type(self), filename)

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of AtomicBasis"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["et_params"] = self.et_params

        if hasattr(self, 'strategy'):
            if isinstance(self.strategy, Strategy):
                d["strategy"] = self.strategy.as_dict()
                d["config"] = self.config
                d["done_setup"] = self._done_setup
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates an AtomicBasis from MSONable dictionary"""
        basis = Basis.from_dict(d)
        element = basis._molecule.name[:-5]
        charge = basis._molecule.charge
        mult = basis._molecule.multiplicity

        instance = cls(name=element, charge=charge, mult=mult)
        instance.results = basis.results
        instance.opt_results = basis.opt_results
        instance._tests = basis._tests
        instance._molecule = basis._molecule
        instance.et_params = d.get("et_params", None)
        instance.strategy = d.get("strategy", None)
        if instance.strategy:
            instance._done_setup = d.get("done_setup", False)
            instance.config = d.get("config", {})
        return instance

    @property
    def element(self) -> MDElement:
        return self._element

    @element.setter
    def element(self, name: str):
        try:
            self._element = MDElement(name.title())
            self._symbol = name.lower()
            self._molecule.name = name + '_atom'
            self._molecule._atom_names = [name]
            self._molecule._coords = [np.array([0.0, 0.0, 0.0])]
            self.results.name = self._molecule.name
        except Exception:
            bo_logger.error("Please enter a valid element")

    @property
    def charge(self) -> int:
        return self._charge

    @charge.setter
    @needs_element
    def charge(self, new_charge: int):
        nelec = self._element.electrons - new_charge
        if nelec < 1:
            bo_logger.warning("A charge of %d would remove all electrons, setting to 0", new_charge)
            self._charge = 0
        else:
            self._charge = new_charge
        self._molecule.charge = self._charge

    @property
    def multiplicity(self) -> int:
        return self._multiplicity

    @multiplicity.setter
    @needs_element
    def multiplicity(self, new_mult: int):
        if (new_mult < 1) or (new_mult - 1 > self._element.electrons):
            bo_logger.warning("Multiplicity can't be set to %d, setting to 1", new_mult)
            self._multiplicity = 1
        else:
            self._multiplicity = new_mult
        self._molecule.multiplicity = self._multiplicity

    @property
    def config(self) -> zt.Configuration:
        return self._config

    @config.setter
    def config(self, new_config: zt.Configuration):
        minimal = self.minimal()
        if zt.compare(minimal, new_config) < 0:
            bo_logger.warning("Configuration %s is insufficient, using minimal config", new_config)
            self._config = minimal
        else:
            self._config = new_config

    def minimal(self) -> zt.Configuration:
        """Returns the minimal basis configuration for this atom"""
        if self._element is None:
            return {}
        return zt.minimal(self._element)

    @needs_element
    def configuration(self, quality: str = 'dz'):
        """Sets the basis set configuration to a desired quality

        Arguments:
             quality (str): name of quality type, see zetatools for options
        """
        try:
            zeta_func = zt.QUALITIES[quality.lower()]
            self._config = zeta_func(self._element)
            config_string = zt.config_to_string(self._config)
            bo_logger.info("Contracted configuration of %s", config_string)
        except KeyError:
            bo_logger.warning(
                "Could not find a %s strategy for configuration, using minimal", quality
            )
            self._config = self.minimal()

    def as_xyz(self) -> str:
        """Returns the atom as an xyz file string"""
        return self._molecule.to_xyz()

    @needs_element
    def setup(
        self,
        method: str = 'ccsd(t)',
        quality: str = 'dz',
        strategy: Strategy = Strategy(),
        reference: tuple[str, Optional[InternalBasis]] = ('cc-pvqz', None),
        params: dict[str, Any] = {},
    ):
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
             self.strategy (Strategy): optimization strategy
             self.config (Configuration): basis set configuration
             self._done_setup (bool): cannot call optimize until this flag is True
        """
        # get configuration
        self.configuration(quality=quality)
        bo_logger.info("Using the %s building strategy", strategy.name)
        bo_logger.info("Method: %s", method)

        # Set or compute reference value
        label, value = reference
        self._molecule.method = method
        self.strategy = strategy
        bo_logger.info("Reference type for this strategy is %s", strategy.eval_type)
        if value is None:
            # Compute
            value = 0.0
            if api.which_backend() == 'Empty':
                bo_logger.warning("No backend currently set, can't compute reference value")
            else:
                bo_logger.info(
                    "Calculating reference value using %s and %s/%s",
                    api.which_backend(),
                    method,
                    label,
                )
                self._molecule.basis = fetch_basis(label, self._symbol)
                success = api.run_calculation(
                    evaluate=strategy.eval_type, mol=self._molecule, params=params
                )
                if success != 0:
                    bo_logger.warning("Reference calculation failed")
                else:
                    value = api.get_backend().get_value(strategy.eval_type)
        self._molecule.add_reference(strategy.eval_type, value)
        bo_logger.info("Reference value set to %f", value)

        # Make a guess for the primitives
        bo_logger.info("Generating starting guess from %s", strategy.guess.__name__)
        self._molecule.basis[self._symbol] = strategy.guess(self, params=strategy.guess_params)
        self._done_setup = True
        bo_logger.info("Atomic basis setup complete")

    @needs_element
    def set_even_tempered(
        self,
        method: str = 'hf',
        accuracy: float = 1e-5,
        max_n: int = 18,
        max_l: int = -1,
        exact_ref: bool = True,
        params: dict[str, Any] = {},
    ):
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
        if len(self.et_params) == 0:
            # optimize new params
            if exact_ref:
                reference = ('exact', data._ATOMIC_HF_ENERGIES[self._element.atomic_number])
            else:
                reference = ('cc-pV5Z', None)
            strategy = EvenTemperedStrategy(max_n=max_n, max_l=max_l)
            self.setup(method=method, strategy=strategy, reference=reference, params=params)
            self.optimize(algorithm='Nelder-Mead', params=params)
            self.et_params = strategy.shells
        else:
            self._molecule.basis[self._symbol] = even_temper_expansion(self.et_params)

    @needs_element
    def optimize(self, algorithm: str = 'Nelder-Mead', params: dict[str, Any] = {}) -> OptResult:
        """Runs the basis optimization

        Arguments:
             algorithm (str): optimization algorithm to use, see scipy.optimize for options
             params (dict): dictionary of parameters to pass to the backend -
                 see the relevant Wrapper object for options

        Returns:
              opt_results (OptResult): a dictionary of scipy results from each opt step
        """
        if self._done_setup:
            self.opt_results = optimize(
                self._molecule, algorithm=algorithm, strategy=self.strategy, **params
            )
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results

    def contract(self):
        """Handles contraction of primitives"""
        raise NotImplementedError
