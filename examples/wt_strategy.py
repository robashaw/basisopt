import basisopt as bo

bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

from basisopt.basis.atomic import AtomicBasis
from basisopt.bse_wrapper import fetch_basis

ne = AtomicBasis('Ne')
ne.set_well_tempered(method='scf', accuracy=1e-4)
print(ne.wt_params)
print(ne._molecule.get_result('energy'), ne._molecule.get_reference('energy'))
for i, p in enumerate(ne.get_basis()['ne']):
    print(f"l = {i}:", p.exps)
ne.save('tmp/neon-wt.obj')
