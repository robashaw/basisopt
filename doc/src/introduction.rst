.. basisopt intro file

.. _`sec:quickstart`:

=============
Quick start
=============  

The basic components in a BasisOpt workflow are: 

- calculation backend
- molecules
- basis sets
- optimization strategies

The tutorials section gives more detailed explanations on how to perform various types of basis set optimization. This quick start guide is intended to go through the steps necessary to get ready for an opimization. 

Load a backend
==============

Currently there are two options for quantum chemistry backend, although any new wrappers will follow the same structures. The first steps upon importing BasisOpt should always be

1. select a backend
2. set the scratch directory (defaults to current directory)

Optionally you may also want to

3. change the logging settings  
4. enable/disable parallel calculations

For Psi4, the python API is used:

.. code-block:: python 

	import basisopt as bo
	bo.set_backend('psi4')
	bo.set_tmp_dir('/tmp/')
	
It is possible to use the Molpro backend using a python API provided by pymolpro. Please see the :ref:`sec:molpro` documentation for further details.

For ORCA (and other non-native backends), we need to give the path to the *directory containing the exexcutables*, e.g.

.. code-block:: python

	bo.set_backend('orca', path='/usr/local/bin/orca/')
	bo.set_tmp_dir('workdir/')

Internally, this will check that the backend is usable. If this is successful, you will get confirmation::

    2022-09-11 21:22:06,682 - orca - INFO - ORCA install dir at: /usr/local/bin/orca
    2022-09-11 21:22:06,683 - set_backend - INFO - Backend set to Orca
	
Otherwise you will get an error::

    2022-09-11 21:23:17,502 - psi4 - ERROR - Psi4 backend not found!
    2022-09-11 21:23:17,502 - set_backend - INFO - Backend set to Dummy

The "Dummy" backend allows BasisOpt to be used for analysis/vizualisation, but cannot perform calculations. 

To change the level of logging (``logging.INFO`` by default), and enable parallelism using DASK (disabled by default):

.. code-block:: python

	import logging
	bo.set_logger(level=logging.WARNING, filename="bo.log")
	bo.set_parallel(True)


Create a molecule
=================

Molecules have four basic properties, before a basis set is added:

- name 
- charge (default 0)
- spin multiplicity (default 1)
- coordinates

Having a unique ``name`` field is important when running calculations over multiple molecules, as it acts as an identifier. 

There are three routes to creating a molecule. 

1. create an empty molecule and add atoms manually

.. code-block:: python

	m = bo.Molecule(name="Chlorine", charge=-1, mult=1)
	m.add_atom(element='Cl', coord=[0., 0., 0.])

2. load from an XYZ file

.. code-block:: python

	m = bo.Molecule.from_xyz("water.xyz", name="Water",
							 charge=0, mult=1)

3. create a diatomic from a string (e.g. nitric oxide with a bond distance of 1.3 angstrom)

.. code-block:: python
	
	from basisopt.molecule import build_diatomic
	m = build_diatomic("NO,1.3", charge=0, mult=2)
	
	
Dummy atoms
===========

You can set certain atoms to be dummies (or ghosts) by specifying the indices of the atoms you want to have no electrons:

.. code-block:: python

	# the first three atoms will be marked as dummies
	m.set_dummy_atoms([0, 1, 2])
	
This can be useful for example in the development of midbond functions, where you might wish to calculate interaction energies. 

	
Add a basis set
===============

Basis sets internally in BasisOpt are dictionaries with the following structure:

.. code-block:: python

	basis = {
		'H': [s-Shell, p-Shell, ...],
		'O': [s-Shell, p-Shell, ...],
		...
	}

where ``basisopt.containers.Shell`` objects have three properties:

- angular momentum ('s', 'p', 'd', ...)
- exponents (numpy array)
- coefficients (list of numpy arrays with same length as exponents)

These can be created manually, fetched directly from the basis set exchange, or read in from file using the basis set exchange API. For example:

.. code-block:: python

	# fetch from BSE library
	m.basis = bo.fetch_basis('cc-pvdz', ['H', 'O'])
	
	# load from file
	import basis_set_exchange as bse
	from basisopt.bse_wrapper import bse_to_internal
	bse_basis = bse.read_formatted_basis_file('vdz.basis', basis_fmt='molpro')
	m.basis = bse_to_internal(bse_basis)
	
	
Add ECPs
========

If you want ECPs to be used on particular atoms, this can be specified by providing a dictionary of ECP names:

.. code-block:: python
	
	m.set_ecps({
		'Br': 'aug-cc-pvtz-pp'
	})
	
If you are using Psi4 or Molpro, these are looked up from the basis set exchange, so should match names given there.
If you are using Orca, the internal Orca ECPs are used, a list of names for which can be found in Section 6.3.3 of the Orca manual. 


Running a calculation
=====================

To test that everything is set up correctly, you can run a quick calculation as follows:

.. code-block:: python
	
	m.method = 'hf' 
	success = bo.run_calculation(evaluate='energy', mol=m)
	print(bo.get_backend().get_value('energy'))

.. toctree::
   :hidden:
