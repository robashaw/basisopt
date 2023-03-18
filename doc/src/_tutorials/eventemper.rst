
.. _`sec:eventemper`:

================================
Even-tempered basis optimization
================================

This tutorial walks through using BasisOpt to optimize an even-tempered basis set for neon, at the SCF level (so not including correlating functions). An example script can also be found in ``examples/et_strategy.py``


Reference calculation
---------------------

For an atomic basis like this, we only really need a single energy value for the method we're using. We could do this in a number of ways: 

- taking a guess at an energy that will always be lower than the true energy
- using a literature value
- running a calculation with an existing large basis set 

In this instance, as we are calculating the Hartree-Fock energy, we will just take the complete basis set limit value for neon, as calculated numerically by Froese-Fischer and coworkers. For atoms, these data are available as part of the library. 

.. code-block:: python

	import basisopt.data
	key = data.atomic_number('ne')
	neon_cbs_energy = data._ATOMIC_HF_ENERGIES[key]
	
For an example of how to perform a calculation yourself, see the quick start guide. 

Optimization strategy
---------------------

An even-tempered basis for neon, at the HF-level, will comprise two shells: `s` and `p`. For each of these we need to optimise a prefactor, and a ratio. If we wish to generate a set that gets to within some error, :math:`\epsilon`, of the CBS limit, we additionally will want to optimise the number of exponents to include in the even-tempered expansion. Internally, all of this can be handled with the ``EvenTemperStrategy``, which can be called directly from an ``AtomicBasis`` object with no additional steps. 

.. code-block:: python

	import basisopt as bo
	from basisopt.basis import AtomicBasis
	
	# initialise backend
	bo.set_backend('orca', path='Applications/orca')
	bo.set_tmp_dir('tmp/')
	
	# optimise basis to within 1e-4 of CBS limit
	ne = bo.AtomicBasis('Ne')
	ne.set_even_tempered(method='rhf', accuracy=1e-4) 
	
	# save basis
	bo.write_json('neon-et.json', ne)

Internally, this is doing the following simple steps:

1. Retrieving reference CBS energy, selecting initial parameters
2. Initial optimisation of the prefactor and ratio for each shell 
3. While energy delta is greater than threshold (and number of functions is less than maximum):

	- increment number of functions in each shell
	- reoptimise prefactor and ratio for each shell

The optimization by default will be carried out using the Nelder-Mead simplex algorithm and with the default backend settings. To change settings, we can pass a dictionary of parameters. See the backend/optimizer documentation for details of valid parameters.  

.. code-block:: python

	ne.set_even_tempered(method='rhf', accuracy=1e-4, params={...})


Results
-------

The results of this optimization will be a set of even-tempered parameters, which can be viewed as follows:

.. code-block:: python

	for i, p in enumerate(ne.et_params):
		print(f"l = {i}:", p)
	print("Optimized energy:", ne.get_result('energy'))
	print("CBS energy:", ne.get_reference('energy'))
	
the results of which should look very similar to this::

	l = 0: (0.24835340569330638, 2.2879816308129683, 18)
	l = 1: (0.1308150091666356, 2.2067231425992575, 13)
	Optimized energy: -128.547048109
	CBS energy: -128.54705

That is, we have an even tempered expansion of the form :math:`0.248(2.288)^{n=0...17}` for the `s`-shell, and :math:`0.131(2.207)^{n=0..12}` for the `p`-shell. This gives an energy well within the 1e-4 threshold (delta is roughly 2e-6). 


.. toctree::
   :hidden:
