import basisopt as bo
from basisopt.basis.molecular import MolecularBasis
from basisopt.opt.strategies import Strategy
from basisopt.testing import PropertyTest

#bo.set_backend('orca', path="/Applications/orca")
bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')
m = bo.Molecule(name="water")
m.from_xyz("water.xyz")

mb = MolecularBasis(name="single")
mb.add_molecule(m)
params = {
    'functional': "b3lyp",
#    'command_line': "TightSCF"
}
strategy = Strategy()
strategy.params = params
mb.setup(method='dft', quality='dzp', strategy=strategy, params=params)
mb.optimize()

energy_test = PropertyTest('EnergyTest', prop='energy')
dipole_test = PropertyTest('DipoleTest', prop='dipole')
mb.register_test(energy_test)
mb.register_test(dipole_test)
mb.run_all_tests(params=params, reference_basis='cc-pvdz')

mb.save("water.pkl")
