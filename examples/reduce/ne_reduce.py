import basisopt as bo
from basisopt.basis.atomic import AtomicBasis
from basisopt.bse_wrapper import fetch_basis
from basisopt.opt.reduce import ReduceStrategy

bo.set_backend('psi4')
bo.set_tmp_dir('/home/me/scr/')

ne = AtomicBasis('Ne')
basis_name = 'Partridge Uncontracted 3'
start_basis = fetch_basis(basis_name, ['Ne'])

# Crank the accuracy
params = {'d_convergence': "1e-8", 'e_convergence': "1e-8", 'scf_type': "pk"}

strategy = ReduceStrategy(
    start_basis, eval_type='energy', method='scf', target=5e-1, shell_mins=[10, 5]
)
ne.setup(method='scf', strategy=strategy, params=params, reference=(basis_name, None))

# If you want to print the guess basis
print(bo.bse_wrapper.internal_basis_converter(start_basis, fmt='molpro'))

# Compute errors and ranks
from basisopt.testing.rank import rank_primitives

errors, ranks = rank_primitives(ne)
print(errors)
print(ranks)

# Actually reduce the basis
res = ne.optimize()
bo.write_json("neon-reduce.json", ne)
