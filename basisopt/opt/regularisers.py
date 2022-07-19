# regularisers, needs expanding
import numpy as np
from typing import Callable

Regulariser = Callable[[np.ndarray], float]

def l1_norm(x: np.ndarray):
    return np.sum(np.abs(x))
    
def l2_norm(x: np.ndarray):
    return np.linalg.norm(x)
    
def linf_norm(x: np.ndarray):
    return np.amax(np.abs(x))