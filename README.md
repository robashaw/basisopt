# BasisOpt

BasisOpt is a python library for the optimization of molecular Gaussian basis sets as used in most quantum chemistry packages. This development version has been forked from the [original version](https://github.com/robashaw/basisopt). 

## Installation

The library is pip installable:

	pip install basisopt

To install a development version instead, clone this repo locally, change to the folder, and use `poetry`. This will cleanly create a virtual environment for the project, that will ensure any changes are publishable. 

	poetry install -v 

You can then start an interactive shell or run a script using 

	poetry run [python3 or script.py]

This is local to the directory, and will not make BasisOpt available elsewhere. You can alternatively create a fresh conda env or pyenv and in the top directory run

	 pip install -e .

However, if you intend to submit a PR for your change, please test building with `poetry` first. 

## Contributing

Contributions are welcomed, either in the form of raising issues or pull requests on this repo. Please take a look at the Code of Conduct before interacting, which includes instructions for reporting any violations.

## Differences from Robert Shaw's repo

The following major changes have been made relative to Robert's original version of the repo.

- Support for a Molpro backend using pymolpro

## Documentation

For dependencies, detailed installation instructions, and a guide to getting started, please refer to the main documentation (currently Robert's original version) [here](https://basisopt.readthedocs.io/en/latest/index.html).

You can build the development version docs locally using sphinx from the ``doc`` directory:

	sphinx-build -b html src build

then open ``index.html`` in the resulting build directory.

## Examples

There are working examples in the examples folder, and these are (or will be) documented in the documentation. 

## Acknowledging usage

If you use this library in your program and find it helpful, that's great! Any feedback would be much appreciated. If you publish results using this library, please consider citing the following paper detailing version 1.0.0:

[J. Chem. Phys. 159, 044802 (2023)](https://doi.org/10.1063/5.0157878)
