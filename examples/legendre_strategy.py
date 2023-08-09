import basisopt as bo

bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

from basisopt.basis.atomic import AtomicBasis

ne = AtomicBasis('Ne')
ne.set_legendre(method='scf', accuracy=1e-4)
print(ne.leg_params)
for i, p in enumerate(ne.get_basis()['ne']):
    print(f"l = {i}:", p.exps)
	