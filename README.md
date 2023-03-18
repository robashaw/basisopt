# BasisOpt

[![Build Status](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)]()
[![codecov](https://codecov.io/gh/robashaw/basisopt/branch/main/graph/badge.svg?token=V8zNdEgBKj)](https://codecov.io/gh/robashaw/basisopt)
[![Documentation Status](https://readthedocs.org/projects/basisopt/badge/?version=latest)](https://basisopt.readthedocs.io/en/latest/?badge=latest)
[![Code Quality](https://api.codiga.io/project/32104/status/svg)](https://app.codiga.io/hub/project/32104/basisopt)

BasisOpt is a python library for the optimization of Gaussian basis sets as used in most quantum chemistry packages. It is currently under development, but should be reasonably stable. 

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

## Coming soon(ish)

- Standard optimization strategies (PCSeg and correlation-consistent)

## Documentation

For dependencies, detailed installation instructions, and a guide to getting started, please refer to the main documentation [here](https://basisopt.readthedocs.io/en/latest/index.html).

## Examples

There are working examples in the examples folder, and these are (or will be) documented in the documentation. 

## Acknowledging usage

If you use this library in your program and find it helpful, that's great! Any feedback would be much appreciated. If you publish results using this library, please consider citing the following paper detailing the implementation:

[ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/640f48e3b5d5dbe9e832e997)
