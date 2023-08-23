# Calculates the HF/cc-pVDZ-PP energy using Molpro
import basisopt as bo

bo.set_backend('molpro')
bo.set_tmp_dir('tmp/')

m = bo.Molecule(name="Krypton")
m.add_atom(element='Kr', coord=[0., 0., 0.])

m.basis = bo.fetch_basis('cc-pvdz-pp', ['Kr'])
m.set_ecps({'Kr': 'cc-pvtz-pp'})

m.method = 'hf'
success = bo.run_calculation(evaluate='energy', mol=m)
print(bo.get_backend().get_value('energy'))