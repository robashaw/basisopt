import basisopt as bo
from basisopt.basis.molecular import MolecularBasis
from basisopt.opt.strategies import Strategy
from basisopt.util import bo_logger

bo.set_backend('psi4')
bo.set_tmp_dir('/Users/ch1jgh/scr/')

mb = MolecularBasis(name="double")
list_of_mols = ['water', 'methane', 'methanol', 'formaldehyde', 'oxygen']
mol_objs = []
for molecule in range(len(list_of_mols)):
	mol_objs.append(bo.molecule.Molecule.from_xyz(list_of_mols[molecule]+'.xyz', 
           name=list_of_mols[molecule]))
	mb.add_molecule(mol_objs[molecule])
    
params = {
    'functional': "wb97x-d",
    'scf_type': "pk"
}

strategy = Strategy()
strategy.params = params
strategy.guess_params = {'name': 'def2-svp'}
mb.setup(method='dft', strategy=strategy, params=params, reference='def2-qzvp')
basis = mb.get_basis()
basis = bo.basis.uncontract(basis)

mb.optimize()
e_opt = []
e_opt.append(strategy.last_objective)
e_diff = e_opt[0]
conv_crit = 1.0e-6
counter = 0

while e_diff > conv_crit:
    bo_logger.info("Starting consistency iteration %d", counter+1)
    mb.optimize()
    e_opt.append(strategy.last_objective)
    e_diff = strategy.delta_objective
    bo_logger.info("Objective function difference from previous iteration: %f\n", e_diff)
    counter += 1

filename = "opt_basis.txt"
bo_logger.info("Writing optimized basis to %s", filename)
f = open(filename, "x")
f.write(bo.bse_wrapper.internal_basis_converter(mb.get_basis(), fmt='molpro'))
f.close()
