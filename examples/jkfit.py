import basisopt as bo

bo.set_tmp_dir('tmp/')
bo.set_backend('orca', path='/Applications/orca')
from basisopt.basis.jkfit import jkfit_collection

guess = bo.fetch_basis('cc-pv5z-jkfit', ['C', 'H'])
guess['li'] = guess['c']
del guess['c']

vtz = bo.fetch_basis('cc-pvdz', ['Li', 'H'])
vqz = bo.fetch_basis('cc-pvtz', ['Li', 'H'])
v5z = bo.fetch_basis('cc-pvtz', ['Li', 'H'])
pairs = [(v5z, None), (vqz, [10, 7, 5, 3, 2, 1, 0]), (vtz, [10, 7, 5, 2, 1, 0, 0])]

jk_bases = jkfit_collection('li', guess, basis_pairs=pairs)
for name, jk in zip(['v5z', 'vqz', 'vtz'], jk_bases):
    bo.write_json(f"li-{name}-jkfit.json", jk)
