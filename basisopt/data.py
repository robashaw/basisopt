# data
from functools import cache

import numpy as np
from mendeleev import element as md_element

# Conversion factors
TO_CM = 219474.63067
TO_EV = 27.2113839
TO_BOHR = 1.88973
TO_ANGSTROM = 0.5291761
FORCE_MASS = 1822.88853


@cache
def atomic_number(element: str) -> int:
    el = md_element(element)
    return el.atomic_number


"""Dictionary converting letter-value angular momenta to l quantum number"""
AM_DICT = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'j': 7, 'k': 8, 'l': 9}

"""Dictionary converting back from l quantum number to letter value"""
INV_AM_DICT = dict((v, k) for k, v in AM_DICT.items())

"""Dictionary with pre-optimised even-tempered expansions for atoms"""
_EVEN_TEMPERED_DATA = {}

ETParams = list[tuple[float, float, int]]

"""Dictionary with pre-optimised well-tempered expansions for atoms"""
_WELL_TEMPERED_DATA = {}

WTParams = list[tuple[float, float, float, float, int]]


def get_even_temper_params(atom: str = 'H', accuracy: float = 1e-5) -> ETParams:
    """Searches for the relevant even tempered expansion
    from _EVEN_TEMPERED_DATA
    """
    if atom in _EVEN_TEMPERED_DATA:
        log_acc = -np.log10(accuracy)
        index = max(4, log_acc) - 4
        index = int(min(index, 3))
        return _EVEN_TEMPERED_DATA[atom][index]
    else:
        return []


def get_well_temper_params(atom: str = 'H', accuracy: float = 1e-5) -> WTParams:
    """Searches for the relevant well tempered expansion
    from _WELL_TEMPERED_DATA
    """
    if atom in _WELL_TEMPERED_DATA:
        log_acc = -np.log10(accuracy)
        index = max(4, log_acc) - 4
        index = int(min(index, 3))
        return _WELL_TEMPERED_DATA[atom][index]
    else:
        return []


"""Essentially exact numerical Hartree-Fock energies for all atoms
   in Hartree. Ref: Saito 2009, doi.org/10.1016/j.adt.2009.06.001"""
_ATOMIC_HF_ENERGIES = {
    1: -0.5,
    2: -2.86167999561,
    3: -7.43272693073,
    4: -14.5730231683,
    5: -24.5290607285,
    6: -37.688618963,
    7: -54.4009342085,
    8: -74.80939847,
    9: -99.4093493867,
    10: -128.547098109,
    11: -161.858911617,
    12: -199.614636425,
    13: -241.876707251,
    14: -288.854362517,
    15: -340.718780975,
    16: -397.504895917,
    17: -459.482072393,
    18: -526.817512803,
    19: -599.164786767,
    20: -676.758185925,
    21: -759.735718041,
    22: -848.405996991,
    23: -942.884337738,
    24: -1043.35637629,
    25: -1149.86625171,
    26: -1262.4436654,
    27: -1381.41455298,
    28: -1506.87090819,
    29: -1638.96374218,
    30: -1777.84811619,
    31: -1923.26100961,
    32: -2075.35973391,
    33: -2234.23865428,
    34: -2399.8676117,
    35: -2572.44133316,
    36: -2752.05497735,
    37: -2938.35745426,
    38: -3131.54568644,
    39: -3331.68416985,
    40: -3538.99506487,
    41: -3753.59772775,
    42: -3975.54949953,
    43: -4204.78873702,
    44: -4441.53948783,
    45: -4685.88170428,
    46: -4937.92102407,
    47: -5197.6984731,
    48: -5465.13314253,
    49: -5740.16915577,
    50: -6022.93169531,
    51: -6313.48532075,
    52: -6611.78405928,
    53: -6917.98089626,
    54: -7232.13836387,
    55: -7553.93365766,
    56: -7883.54382733,
    57: -8221.0667026,
    58: -8566.87268128,
    59: -8921.18102813,
    60: -9283.88294453,
    61: -9655.09896927,
    62: -10034.9525472,
    63: -10423.5430217,
    64: -10820.6612101,
    65: -11226.5683738,
    66: -11641.4525953,
    67: -12065.2898028,
    68: -12498.1527833,
    69: -12940.1744048,
    70: -13391.4561931,
    71: -13851.8080034,
    72: -14321.2498119,
    73: -14799.812598,
    74: -15287.5463682,
    75: -15784.5331876,
    76: -16290.6485954,
    77: -16806.1131497,
    78: -17331.0699646,
    79: -17865.4000842,
    80: -18408.9914949,
    81: -18961.8248243,
    82: -19524.0080381,
    83: -20095.5864271,
    84: -20676.500915,
    85: -21266.8817131,
    86: -21866.7722409,
    87: -22475.8587125,
    88: -23094.3036664,
    89: -23722.1920622,
    90: -24359.622444,
    91: -25007.1098723,
    92: -25664.3382676,
    93: -26331.4549589,
    94: -27008.7194421,
    95: -27695.8872166,
    96: -28392.7711729,
    97: -29099.8316144,
    98: -29817.418916,
    99: -30544.9721855,
    100: -31282.777599,
    101: -32030.9329688,
    102: -32789.5121404,
    103: -33557.9504126,
    104: -34336.6215955,
    105: -35125.5446447,
    106: -35924.7569387,
    107: -36734.3244057,
    108: -37554.1214298,
    109: -38384.3424294,
    110: -39225.1624771,
    111: -40076.3544159,
    112: -40937.7978561,
    113: -41809.5353119,
    114: -42691.6571511,
    115: -43584.1991337,
    116: -44487.1002441,
    117: -45400.4748133,
    118: -46324.3558151,
}
