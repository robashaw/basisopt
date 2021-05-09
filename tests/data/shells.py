from basisopt.containers import Shell
import numpy as np

_nsexp = 4
_nsfuncs = 2
_h_vdz_exps_s = [13.01, 1.962, 0.4446, 0.122]
_h_vdz_coefs_s = [
    [0.019685, 0.137977, 0.478148, 0.50124],
    [0.0, 0.0, 0.0, 1.0]
]

_npexp = 1
_npfuncs = 1
_h_vdz_exps_p = [0.727]
_h_vdz_coefs_p = [[1.0]]

_compute_values = [
    ([0.0, 0.0, 0.0]  , [0.320755882986,  0.0]),
    ([0.5, 0.0, 0.0]  , [0.281891520077,  0.20370077325]),
    ([0.0, 0.5, 0.0]  , [0.281891520077, -0.0]),
    ([0.0, 0.0, 0.5]  , [0.281891520077,  0.20370077325]),
    ([-1.0, 0.0, 0.5] , [0.202122773036, -0.2201636372]),
    ([2.0, 0.2, -2.0] , [0.056801640967,  0.0039896706]),
    ([5.0, 1.0, -0.5] , [0.005750421985,  1.2648422397e-08])
]

def get_vdz_internal():
    s_shell = Shell()
    s_shell.l = 's'
    s_shell.exps = np.array(_h_vdz_exps_s)
    s_shell.coefs = [np.array(c) for c in _h_vdz_coefs_s]
    
    p_shell = Shell()
    p_shell.l = 'p'
    p_shell.exps = np.array(_h_vdz_exps_p)
    p_shell.coefs = [np.array(c) for c in _h_vdz_coefs_p]
    
    return {'h': [s_shell, p_shell]}
    
def shells_are_equal(s1, s2):
    equal = (s1.l == s2.l)
    equal &= (np.sum(np.abs(s1.exps - s2.exps)) == 0)
    for c1, c2 in zip(s1.coefs, s2.coefs):
        equal &= ((np.sum(np.abs(c1 - c2))) == 0)
    return equal