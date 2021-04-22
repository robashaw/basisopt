# preconditioners, needs expanding
import numpy as np

def inverse(inv_func):
    """Decorator that adds an inverse function as an attribute
       All preconditioners must be decorated with an inverse,
       which should usually have the same signature as the parent.
    
       Arguments:
            inv_func (func): the inverse of the preconditioner
    """
    def decorator(func):
        func.inverse = inv_func
        return func
    return decorator
    
def _unit_inverse(y, minval=1e-4, ratio=1.4):
    """Default, does nothing"""
    return y
    
@inverse(_unit_inverse)
def unit(x):
    """Identity function"""
    return x

@inverse(_unit_inverse)
def make_positive(x, minval=1e-4, ratio=1.4):
    """Returns x with all values >= minval
       If multiple values are < minval, the new values
       will be minval * (ratio**n)
    """
    y = np.copy(x)
    for ix, v in enumerate(y):
        if v < minval:
            y[ix] = minval
            minval *= ratio
    return y
    
def _logistic_inverse(y, minval=1e-4, maxval=1e5, alpha=1.0, x0=0.0):
    """Inverse for logistic function"""
    x = (y - minval)/maxval
    x = (1.0 / x) - 1.0 
    x = -np.log(x) / alpha
    return (x + x0)
      
@inverse(_logistic_inverse)  
def logistic(x, minval=1e-4, maxval=1e5, alpha=1.0, x0=0.0):
    """Logistic function"""
    y = 1.0 + np.exp(-alpha*(x-x0))
    return (minval + (maxval / y))