# regularisers, needs expanding
from typing import Any

import numpy as np

from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import InternalBasis, Result
from basisopt.molecule import Molecule


class OptRecord(Result):
    """ """

    def __init__(self, name: str = "Empty"):
        super().__init__(name=name)
        self.functional_groups = []
        self.species = []
        self.coords = []
        self.molecule = None
        self.counts = None

    def get_molecule(self) -> Molecule:
        """Returns Molecule from species/coords"""
        if not self.molecule:
            self.molecule = Molecule(name=self.name)
            for at, xyz in zip(self.species, self.coords):
                self.molecule.add_atom(at, xyz)
            self.molecule.basis = self.get_basis()
        return self.molecule

    def get_basis(
        self,
        step_back: int = 0,
        default: str = "cc-pvtz",
        add_atoms: list[str] = ['H'],
        ls: list[str] = ['s', 'p', 'd', 'f'],
    ) -> InternalBasis:
        """Gets the basis from this record with additional terms

        Arguments:
              step_back (int): how many steps back in the record to go
              default (str): name of BSE basis to use as default
              add_atoms (list[str]): list of additional atoms to include
              ls (list[str]): list of ang. momentum shells to include

         Returns:
               an InternalBasis corresponding to this record with any
               additional atoms specified
        """
        unique_atoms = set(self.species)
        for a in add_atoms:
            unique_atoms.add(a.title())
        basis = fetch_basis(default, list(unique_atoms))
        for c in self._children:
            shells = [c.get_data(l, step_back=step_back) for l in ls]
            basis[c.name.lower()] = shells
        return basis

    def get_counts(self) -> dict[str, int]:
        """Returns a dictionary of form
        {atom: no. of atoms in molecule}
        """
        if not self.counts:
            sp_str = "".join(self.species)
            self.counts = {at: sp_str.count(at) for at in set(self.species)}
        return self.counts

    def chemical_formula(self) -> str:
        """Represent molecule as its chemical formula"""
        counts = self.get_counts()
        name = ""
        for k, v in counts.items():
            if v > 1:
                name += f"{k}{v}"
            elif v != 0:
                name += f"{k}"
        return name

    def n_heavy(self) -> int:
        """Returns number of non-hydrogen atoms"""
        counts = self.get_counts()
        n = sum(counts.values())
        if 'H' in counts:
            n -= counts['H']
        return n

    def as_dict(self) -> dict[str, Any]:
        """Converts to MSONable dictionary"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["functional_groups"] = self.functional_groups
        d["species"] = self.species
        d["coords"] = self.coords
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates OptRecord from MSONable dictionary"""
        result = Result.from_dict(d)
        instance = cls(name=result.name)
        instance._data_keys = result._data_keys
        instance._data_values = result._data_values
        instance.depth = result.depth
        instance._children = result._children
        instance.functional_groups = d.get("functional_groups", [])
        instance.species = d.get("species", [])
        coords = d.get("coords", [])
        instance.coords = np.array([c['data'] for c in coords])
        return instance
