import basisopt as bo

bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

from basisopt.basis.atomic import AtomicBasis
from basisopt.bse_wrapper import fetch_basis
from basisopt.util import write_json

ne = AtomicBasis('Ne')
ne.setup(method='scf')
ne.optimize(algorithm='Nelder-Mead')
print(ne._molecule.get_result('energy'), ne._molecule.get_reference('energy'))
# ne.save('tmp/neon-def.obj')
write_json("tmp/neon.json", ne)
