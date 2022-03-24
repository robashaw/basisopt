import basisopt as bo
bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

from basisopt.basis.atomic import AtomicBasis
from basisopt.bse_wrapper import fetch_basis
ne = AtomicBasis('Ne')
ne.set_even_tempered(method='scf', accuracy=1e-4)
print(ne.et_params)
print(ne._molecule.get_result('energy'), ne._molecule.get_reference('energy'))
ne.save('tmp/neon-et.obj')