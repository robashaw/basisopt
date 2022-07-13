
import functools
import logging
import copy
import pickle
from .basis import Basis
from .atomic import AtomicBasis
from basisopt import api
from basisopt.containers import Result
from basisopt.util import bo_logger
from basisopt.bse_wrapper import fetch_basis
from basisopt.exceptions import EmptyBasis, DataNotFound
from basisopt.opt.strategies import Strategy
from basisopt.opt.optimizers import collective_optimize

class MolecularBasis(Basis):
    def __init__(self, name='Empty', molecules=[]):
        """"""
        super(MolecularBasis, self).__init__()
        self.name = name
        self.basis = {}
        self._molecules = {}
        self._atoms = set()
        self._atomic_bases = {}
        self._done_setup = False
        for m in molecules:
            self.add_molecule(m)
        
    def save(self, filename):
        """Pickles the MolecularBasis object into a binary file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()
        bo_logger.info("Dumped object of type %s to %s", type(self), filename)
        
    def add_molecule(self, molecule):
        if molecule.name in self._molecules:
            bo_logger.warning(f"Molecule with name {molecule.name} being overwritten")
        self._molecules[molecule.name] = molecule
        for atom in molecule.unique_atoms():
            self._atoms.add(atom.lower())
    
    def get_molecule(self, name):
        if name in self._molecules:
            return self._molecules[name]
        else:
            bo_logger.warning(f"No molecule with name {name}")
            return None
            
    def get_basis(self):
        return self.basis
        
    def get_atomic_basis(self, atom):
        if atom in self._atomic_bases:
            return self._atomic_bases[atom]
        else:
            return None
    
    def unique_atoms(self):
        return list(self._atoms)
        
    def molecules(self):
        return [m for m in self._molecules.values()]
        
    def run_test(self, name, params={}, reference_basis=None, do_print=True):
        t = self.get_test(name)
        if t is None:
            bo_logger.warning("No test with name %s", name)
        else:
            try:
                child = self.results.get_child(name)
            except DataNotFound:
                new_result = Result(name=name)
                self.results.add_child(new_result)
                child = new_result
                
                # calculate reference values
                bo_logger.info("Calculating reference values for test %s", name)
                str_basis = isinstance(reference_basis, str)
                for m in self.molecules():
                    t.molecule = m
                    if str_basis:
                        t.calculate_reference(m.method, basis_name=reference_basis,
                                              params=params)
                    else:
                        t.calculate_reference(m.method, basis=reference_basis,
                                              params=params)
                    child.add_data(f"{m.name}_ref", t.reference)
                    
            results = {}
            for m in self.molecules():
                t.result = t.calculate(m.method, self.basis, params=params)
                child.add_data(m.name, t.result)
                results[m.name] = t.result
                if do_print:
                    bo_logger.info("%s: %s", m.name, str(t.result))  
            return results  
                
    def run_all_tests(self, params={}, reference_basis=None):
        results = {}
        for t in self._tests:
            results[t.name] = self.run_test(t.name, params=params,
                                            reference_basis=reference_basis,
                                            do_print=False)
        # print results
        header = "Molecule"
        for t in self._tests:
            header += f"\t{t.name}"
        bo_logger.info(header)
        for m in self.molecules():
            res_string = f"{m.name}"
            for k, v in results.items():
                res_string += f"\t{v[m.name]}"
            bo_logger.info(res_string)
            
    
    def setup(self, method='ccsd(t)', quality='dz', strategy=Strategy(), reference='cc-pvqz', params={}):
        if len(self._atoms) == 0:
            raise EmptyBasis
        
        for m in self.molecules():
            m.method = method
        
        self._atomic_bases = {}
        for atom in self._atoms:
            self._atomic_bases[atom] = AtomicBasis(atom)
            bo_logger.info("Doing setup for atom %s", atom)
            self._atomic_bases[atom].setup(method=method, quality=quality, 
                                           strategy=strategy,
                                           reference=('dummy', 0.), params=params)
        self.basis = {k: v.get_basis() for k, v in self._atomic_bases.items()}
        if reference is not None:
            if api.which_backend() == 'Empty':
                bo_logger.warning(f"No backend currently set, can't compute reference value")
            else:
                ref_basis = fetch_basis(reference, self.unique_atoms())
                for m in self.molecules():
                    bo_logger.info("Calculating reference value for molecule %s using %s and %s/%s",
                                 m.name, api.which_backend(), method, reference)
                    m.basis = ref_basis
                    success = api.run_calculation(evaluate=strategy.eval_type, mol=m, params=params)
                    if success != 0:
                        bo_logger.warning("Reference calculation failed")
                        value = 0.
                    else:
                        value = api.get_backend().get_value(strategy.eval_type)         
                    m.add_reference(strategy.eval_type, value)
                    bo_logger.info("Reference value set to %f", value)
        
        self._done_setup = True
        bo_logger.info("Molecular basis setup complete")
        
    def optimize(self, algorithm='Nelder-Mead', params={}, reg=lambda x: 0, npass=1, parallel=False):
        if self._done_setup:
            opt_data = [(k, algorithm, v.strategy, reg, params)
                         for k, v in self._atomic_bases.items()]
            self.opt_results = collective_optimize(self._molecules.values(), self.basis, opt_data=opt_data,
                                               npass=npass, parallel=parallel)
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results
        
        
            
    
    
        
