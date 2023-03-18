.. basisopt install file

.. _`sec:installation`:

============
Installation
============

The easiest way to install is via pip::

	pip install basisopt

Install from Source
-------------------

First, make sure you install the python package `poetry`. Then, clone the repository from `github
<https://github.com/robashaw/basisopt>`_::

    git clone https://github.com/robashaw/basisopt.git
    cd basisopt
    poetry install -v

Then, an interactive shell or script can be run with::

	poetry run [python3 or script.py]
	
Alternatively, if you wish to use your locally-changed package elsewhere, then try the following with `conda`, for example, in the top-level directory::

	conda create -n basisopt python=3.10
	pip install -e .

Dependencies
------------

There are fairly minimal requirements, and versions listed are minimum tested versions. Older versions _may_ work but are untested. The exception is Mendeleev where upstream changes mean we currently require version 0.9; this will hopefully be fixed soon.  

* python >= 3.9
* colorlog >= 4.1 
* numpy >= 1.21.6
* scipy >= 1.8.1
* matplotlib >= 3.3
* pandas >= 1.3.5
* monty >= 2022.4.26
* mendeleev == 0.9.0
* basis_set_exchange >= 0.9

You will also need a quantum chemistry backend. Currently supported codes are:

* Psi4 : version 1.4 or above
* ORCA : version 4.2.x or version 5.x 

There will eventually be a tutorial on how to implement wrappers for different packages. If you do, please consider submitting a pull request so that others may benefit. 
  
Testing and development
-----------------------

We use ``pytest`` for our CI::

    poetry run pytest
	
We also use ``black``, ``isort``, and ``flake8``, for code-formatting and linting. Any PR will automatically be checked using these packages before being considered for merging. 

Documentation
-------------

To build this documentation locally, make sure you have the following packages

* Sphinx >= 2.1.2

In the ``doc`` directory, run the following::
	
	mkdir build
	sphinx-build src/ build/ 

This will by default create HTML; to make other formats, consult the ``sphinx-build`` documentation.

.. toctree::
   :hidden:
