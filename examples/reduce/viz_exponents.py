import basisopt as bo
from basisopt.basis.atomic import AtomicBasis
from basisopt.bse_wrapper import fetch_basis
import matplotlib.pyplot as plt
from basisopt.viz.basis import plot_exponents

ne = bo.read_json("neon-reduce.json")

basis_name = 'Partridge Uncontracted 3'
start_basis = fetch_basis(basis_name, ['Ne'])

start_basis['partridge'] = start_basis['ne']
start_basis['optimized'] = ne.get_basis()['ne']

fig, ax = plot_exponents(start_basis, atoms=['partridge', 'optimized'], split_by_shell=True)
ax[0].set_title('Partridge Uncontracted 3')
ax[1].set_title('(10s5p) optimized')
plt.savefig('basis_plot.png')
