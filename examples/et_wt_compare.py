import basisopt as bo
from basisopt.basis.atomic import AtomicBasis
from basisopt.basis.basis import InternalBasis
from basisopt.util import bo_logger

bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

bo_logger.info("Optimizing even-tempered basis set")
ne_et = AtomicBasis('Ne')
ne_et.set_even_tempered(method='scf', accuracy=1e-4)

bo_logger.info("Optimizing well-tempered basis set")
ne_wt = AtomicBasis('Ne')
ne_wt.set_well_tempered(method='scf', accuracy=1e-4)

bo_logger.info("Visualizing the difference between the sets")
viz_basis = InternalBasis()
viz_basis['even'] = ne_et.get_basis()['ne']
viz_basis['well'] = ne_wt.get_basis()['ne']

import matplotlib.pyplot as plt
from basisopt.viz.basis import plot_exponents

fig, ax = plot_exponents(viz_basis, atoms=['even', 'well'], split_by_shell=True)
ax[0].set_title('Even-tempered')
ax[1].set_title('Well-tempered')

filename = "comparison_plot.png"
bo_logger.info("Writing exponents plot to %s", filename)
plt.savefig(filename)


