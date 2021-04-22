from basisopt import api, data
from basisopt.exceptions import PropertyNotAvailable
from .preconditioners import make_positive
from basisopt.basis.guesses import bse_guess
import numpy as np

class Strategy:
    """ Object to describe and handle basis set optimization strategies. 
        All strategy types should inherit from here, and give a description
        of the approach in the docs. This is a MINIMAL implementation, so all
        methods here should usually be overridden in child classes
    
        This class also acts as a 'Default' optimization strategy. The alg is
        as follows:
        
        --------------------------- ALGORITHM ----------------------------
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: cc-pVDZ
        Pre-conditioner: any (default, make sure exponents are positive)
        
        No initialisation needed.
        Optimize each shell in increasing order of angular momentum,
        so self._step = l+1, ends when self._step = max_l+1
        No iteration by default. 
        ------------------------------------------------------------------
    
        Attributes:
            name (str): identifier
            eval_type (str): property to evaluate
            params (dict): parameters for backend, see relevant Wrapper for options
            guess (func): function to generate starting guess exponents
            guess_params (dict): parameters to pass to guess
            pre (func): function to precondition exponents - must have an inverse attribute
            pre.params (dict): parameters to pass to the preconditioner
    
        Private attributes:
            _step (int): tracks what step of optimization we're on
    """
    def __init__(self, eval_type='energy', pre=make_positive):
        self.name = 'Default'
        self._eval_type = ''
        self.eval_type = eval_type
        self.params = {}
        self.guess = bse_guess
        self.guess_params = {'name': 'cc-pvdz'}
        self._step = -1
        self.pre = pre
        self.pre.params = {}
    
    @property
    def eval_type(self):
        return self._eval_type
    
    def initialise(self, basis, element):
        """ Initialises the strategy (does nothing in default)
            
            Arguments:
                basis: internal basis dictionary
                element: symbol of the atom being optimized
        """
        pass
    
    @eval_type.setter
    def eval_type(self, name):
        wrapper = api.get_backend()
        if name in wrapper.all_available:
            self._eval_type = name
        else:
            raise PropertyNotAvailable(name)
    
    def get_active(self, basis, element):
        """ Arguments:
                 basis: internal basis dictionary 
                 element: symbol of the atom being optimized
        
            Returns:
                 the set of exponents currently being optimised
        """
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)
        
    def set_active(self, values, basis, element):
        """ Sets the currently active exponents to the given values.
            
            Arguments:
                values (list): list of new exponents
                basis: internal basis dictionary
                element: symbol of atom being optimized
        """
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)
        
    def next(self, basis, element, objective):
        """ Moves the strategy forward a step (see algorithm)
        
            Arguments:
                basis: internal basis dictionary
                element: symbol of atom being optimized
                objective: value of objective function from last steps
        
            Returns:
                True if there is a next step, False if strategy is finished
        """
        self._step += 1
        maxl = len(basis[element])-1
        return (maxl != self._step) 
            
    