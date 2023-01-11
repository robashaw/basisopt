import basisopt as bo
bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

from basisopt.basis.atomic import AtomicBasis
from basisopt.bse_wrapper import fetch_basis
from basisopt.opt.reduce import ReduceStrategy

ne = AtomicBasis('Ne')
guess_basis = fetch_basis('cc-pvdz', ['Ne'])
strategy = ReduceStrategy(guess_basis,
                          target=1e-4,
                          shell_mins=[4, 2, 0])
ne.setup(method='scf', strategy=strategy)
ne.optimize(algorithm='Nelder-Mead')
#ne.save('tmp/neon-def.obj')
json_string = ne.to_json()
with open('tmp/neon-reduce.json', 'w') as f:
    f.write(json_string)
