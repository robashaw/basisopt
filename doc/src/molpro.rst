.. basisopt molpro wrapper file

.. _`sec:molpro`:

=========================
Using Molpro as a backend
=========================

The Molpro backend for BasisOpt uses the pymolpro Python library. Both the commercial 
`Molpro quantum chemistry package <https://www.molpro.net>`_ and the free `pymolpro library <https://molpro.github.io/pymolpro/index.html#>`_ 
must be available to use the backend. The backend can then specified in the same manner as for other quantum chemistry packages.

.. code-block:: python 

	bo.set_backend('molpro')
	bo.set_tmp_dir('/tmp/')

This guide is intended to highlight how to specify certain types of Molpro calculation via BasisOpt, including passing Molpro options and
directives.

Density Functional Theory calculations
======================================

In Molpro, Kohn-Sham DFT calculations are requested using either the ``rks`` or ``uks`` commands, for spin-restricted and spin-unrestricted, respectively. The exchange-correlation functional must also be specified. To maintain consistency with other BasisOpt backends, the choice of
functional is specified by setting the ``functional`` params. For example, a PBE0 calculation using the spin-restricted Kohn-Sham program
can be setup as:

.. code-block:: python 

	params = {'functional': "pbe0"}
	strategy = Strategy()
	mb.setup(method='rks', strategy=strategy, params=params)
	
Specifying options and directives
=================================

A `params` block is used to pass options and directives to Molpro on a per-method basis. This is done using a *method*-params key and, as
an example, requesting that no orbitals are treated with a frozen-core approximation in an MP2 calculation can be specified in the 
following way:

.. code-block:: python 

	params = {'mp2-params': ";core,0"}
	strategy = Strategy()
	mb.setup(method='mp2', strategy=strategy, params=params)
	
Note that the ``;`` and ``,`` separators used in Molpro input files must be included as part of the params values.

Post-HF methods
===============

When running a post-Hartree-Fock calculation in Molpro, the reference wavefunction must also be specified. In BasisOpt, the reference
calculation is somewhat *hard-coded* and will be determined from the post-HF method selected. Currently, requesting a method of `mp2`, 
`ccsd` or `ccsd(t)` will use a `hf` reference. The methods `rmp2`, `uccsd` and `uccsd(t)` will use an `rhf` reference, and `ump2` uses
the spin-unrestricted `uhf` reference.

It is possible to pass Molpro options and/or directives to both the reference and the correlated calculation. For example, to increase
the accuracy of the HF reference and also request that no orbitals are treated with a frozen-core approximation in an MP2 calculation:

.. code-block:: python 

	params = {'hf-params': ";accu,14", 'mp2-params': ";core,0"}
	strategy = Strategy()
	mb.setup(method='mp2', strategy=strategy, params=params)

Parameters for an RHF reference would use an ``rhf-params`` key, while a UHF reference would require ``uhf-params``.


.. toctree::
   :hidden:
