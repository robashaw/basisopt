# regularisers, needs expanding
import numpy as np

def l1_norm(x):
    return np.sum(np.abs(x))
    
def l2_norm(x):
    return np.linalg.norm(x)
    
def linf_norm(x):
    return np.amax(np.abs(x))