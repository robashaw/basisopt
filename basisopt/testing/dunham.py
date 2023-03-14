# duham test
from typing import Any, Optional

import numpy as np
from mendeleev import element as md_element

from basisopt import api, data
from basisopt.containers import InternalBasis
from basisopt.exceptions import EmptyCalculation, FailedCalculation
from basisopt.molecule import Molecule, build_diatomic
from basisopt.util import fit_poly

from .test import Test

_VALUE_NAMES = ["Ee", "Re", "BRot", "ARot", "We", "Wx", "Wy", "De", "D0"]

DunhamResults = tuple[np.poly1d, float, np.ndarray]


def dunham(
    energies: np.ndarray,
    distances: np.ndarray,
    mu: float,
    poly_order: int = 6,
    angstrom: bool = True,
    Emax: float = 0,
) -> DunhamResults:
    "Performs a Dunham analysis on a diatomic, given energy/distance values around a minimum and the reduced mass mu"
    # convert units
    An = mu * data.FORCE_MASS
    if angstrom:
        distances *= data.TO_BOHR
    poly_order = max(poly_order, 3)

    # perform polynomial fit to data
    p, xref, re, pt = fit_poly(distances, energies, poly_order)

    # Energy at minimum, first rotational constant, and first vibrational constant
    Ee = pt[0]
    Be = 0.5 * data.TO_CM / (An * re**2)
    We = data.TO_CM * np.sqrt(2.0 * np.abs(pt[2]) / An)

    # Compute normalised derivatives
    npt = [(pt[i + 3] / pt[2]) * re ** (i + 1) for i in range(poly_order - 2)]

    # Second rotational constant
    Ae = -6.0 * Be**2 * (1.0 + npt[0]) / We

    # First anharmonic corrections require n >= 6
    Wexe = 0.0
    Weye = 0.0
    if poly_order > 5:
        Wexe = -1.5 * (npt[1] - 1.25 * npt[0] ** 2) * Be
        Weye = (
            0.5
            * (
                10.0 * npt[3]
                - 35.0 * npt[0] * npt[2]
                - 8.5 * npt[1] ** 2
                + 56.125 * npt[1] * npt[0] ** 2
                - 22.03125 * npt[0] ** 4
            )
            * Be**2
            / We
        )

    # Dissociation energies
    De = 0.0
    D0 = 0.0
    if Emax != 0:
        De = (Emax - Ee) * data.TO_EV
        D0 = De - 0.5 * (We - 0.5 * Wexe) * data.TO_EV / data.TO_CM

    results = np.array([Ee, re * data.TO_ANGSTROM, Be, Ae, We, Wexe, Weye, De, D0])
    return p, xref, results


class DunhamTest(Test):
    """Carries out a Dunham analysis on a diatomic, calculating spectroscopic constants
    Initialised with either a diatomic Molecule object, or a mol_string of the form
    "Atom1Atom2,separation in Ang", e.g. "H2,0.9", "NO,1.2", "LiH,1.3" etc.

    Results:
         returned as numpy array, as well as archived
         Ee:         Energy at eq. separation (Ha)
         Re:         Eq. separation (Ang)
         BRot, ARot: First and second rotational constants (cm-1)
         We:         First vibrational constant (cm-1)
         Wx, Wy:     x and y anharmonic corrections to We (cm-1)
         De:         Dissociation energy (eV)
         D0:         Zero-point dissociation energy (eV)

    Additional data stored:
         StencilRi (numpy array): the separation values (Ang) used in the polynomial fit
         StencilEi (numpy array): the energy values (Ha) at each point in the fit

    Additional attributes:
         poly_order (int): order of polynomial to fit, >= 3
         step (float): step size in Angstrom to use for polynomial fit
         Emax (float): energy in Ha to calculate dissociation from (default 0)
         poly (poly1d): fitted polynomial
         shift (float): the shift for separations used in the polynomial fit
         e.g. to calculate the value at the point R, use poly(R-shift)

    """

    def __init__(
        self,
        name: str,
        mol: Optional[Molecule] = None,
        mol_str: str = "",
        charge: int = 0,
        mult: int = 1,
        poly_order: int = 6,
        step: float = 0.05,
        Emax: float = 0,
    ):
        super().__init__(name, mol=mol, charge=charge, mult=mult)
        self.poly_order = max(3, poly_order)
        self.step = step
        self.Emax = Emax
        self.poly = None
        self.shift = 0.0

        if len(mol_str) != 0:
            self.from_string(mol_str, charge=charge, mult=mult)

        self.reduced_mass()

    def as_dict(self):
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["poly_order"] = self.poly_order
        d["step"] = self.step
        d["Emax"] = self.Emax
        return d

    @classmethod
    def from_dict(cls, d):
        test = Test.from_dict(d)
        instance = cls(
            test.name,
            poly_order=d.get("poly_order", 6),
            step=d.get("step", 0.05),
            Emax=d.get("Emax", 0),
        )
        instance.molecule = test.molecule
        instance.reference = test.reference
        instance._data_keys = test._data_keys
        instance._data_values = test._data_values
        instance._children = test._children
        instance.depth = test.depth
        return instance

    def from_string(self, mol_str: str, charge: int = 0, mult: int = 1):
        """Makes a diatomic molecule from string to use in test

        Arguments:
             mol_str (str): string of diatomic and separation in Angstrom
                            e.g. "NO,1.3", "H2,0.9", "LiH,1.1" etc
             charge (int): overall charge of diatomic
             mult (int): spin multipilicity of diatomic
        """
        self.molecule = build_diatomic(mol_str, charge=charge, mult=mult)

    def reduced_mass(self) -> float:
        """Calculate the reduced mass of the diatomic"""
        atom1 = md_element(self.molecule._atom_names[0].title())
        atom2 = md_element(self.molecule._atom_names[1].title())
        return (atom1.mass * atom2.mass) / (atom1.mass + atom2.mass)

    def calculate(
        self, method: str, basis: InternalBasis, params: dict[str, Any] = {}
    ) -> np.ndarray:
        """Calculates the Dunham analysis results

        Args:
               method (str): calculation method to use
               basis (InternalBasis): orbital basis set
               params (dict): parameters to pass to backend

        Returns:
               array of results in order specified by _VALUE_NAMES
        """
        if not self.molecule:
            raise EmptyCalculation
        self.molecule.basis = basis
        self.molecule.method = method

        # create the stencil
        rvals = np.zeros(self.poly_order + 1)
        # set the midpoint as the current separation
        midix = int(self.poly_order / 2)
        rvals[midix] = self.molecule.distance(0, 1)
        for i in range(midix + 1, self.poly_order + 1):
            rvals[i] = rvals[i - 1] + self.step
        for i in range(midix):
            rvals[midix - i - 1] = rvals[midix - i] - self.step

        # calculate energies
        wrapper = api.get_backend()
        energies = []
        for r in rvals:
            self.molecule._coords[0] = np.array([0.0, 0.0, -0.5 * r])
            self.molecule._coords[1] = np.array([0.0, 0.0, 0.5 * r])
            success = api.run_calculation(evaluate='energy', mol=self.molecule, params=params)
            if success != 0:
                raise FailedCalculation
            energies.append(wrapper.get_value('energy'))

        # perform the analysis
        energies = np.array(energies)
        self.poly, self.shift, results = dunham(
            energies, rvals, self.reduced_mass(), poly_order=self.poly_order, Emax=self.Emax
        )

        # store results
        self.add_data("StencilRi", rvals * data.TO_ANGSTROM)
        self.add_data("StencilEi", energies)
        for n, r in zip(_VALUE_NAMES, results):
            self.add_data(n, r)

        return results
