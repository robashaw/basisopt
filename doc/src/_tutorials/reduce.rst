:orphan:
.. _`sec:reduce`:

=====================================
Systematic removal of basis functions
=====================================

This example demonstrates how BasisOpt can be used to take a large, near-saturated basis set and reduce it in size until it reaches specific criteria. More specifically, the (18s13p) basis set of Partridge is obtained from the Basis Set Exchange, and functions are removed along with exponent re-optimization until a basis composition of (10s5p) is reached. The full example script can be found in ``examples/reduce/ne_reduce.py``.

Loading a basis from the Basis Set Exchange
-------------------------------------------

The `Basis Set Exchange <https://www.basissetexchange.org>`_ is a well-known repository of basis sets that includes a Python interface. BasisOpt uses a wrapper to fetch the basis sets from the Exchange, with the following code example fetching the `Partridge Uncontracted 3 <https://aip.scitation.org/doi/10.1063/1.456157>`_ basis for neon. 

.. code-block:: python

	from basisopt.bse_wrapper import fetch_basis
	ne = AtomicBasis('Ne')
	basis_name = 'Partridge Uncontracted 3'
	start_basis = fetch_basis(basis_name, ['Ne'])
	
Optimization strategy
---------------------

In this example we have chosen to use the supplied ReduceStrategy. This automates the removal of functions by computing the incremental change in energy if each individual function was to be removed, and using this to rank the functions for removal. The remaining exponents are then re-optimized until a user-supplied threshold for the incremental energy change is reached. A minimum basis set composition can also be requested. In this case the target is a specific composition, hence the energetic threshold is artificially high. We elected to do this at the HF-SCF level of theory.

.. code-block:: python

	from basisopt.opt.reduce import ReduceStrategy
	strategy = ReduceStrategy(start_basis,
	                          eval_type='energy', method='scf',
	                          target=5e-1, shell_mins=[10, 5])
	ne.setup(method='scf', strategy=strategy, params=params, reference=(basis_name, None))
	res=ne.optimize()

Saving to json format
---------------------

Finally, we save the optimization to json format for later reuse.

.. code-block:: python

	bo.write_json("neon-reduce.json", ne)

.. toctree::
   :hidden:
