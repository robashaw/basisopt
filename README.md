# BasisOpt

[![Build Status](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)]()
[![codecov](https://codecov.io/gh/robashaw/basisopt/branch/main/graph/badge.svg?token=V8zNdEgBKj)](https://codecov.io/gh/robashaw/basisopt)
[![Documentation Status](https://readthedocs.org/projects/basisopt/badge/?version=latest)](https://basisopt.readthedocs.io/en/latest/?badge=latest)
[![Code Quality](https://api.codiga.io/project/32104/status/svg)]()

BasisOpt is a python library for the optimization of Gaussian basis sets as used in most quantum chemistry packages. It is currently under development, but should be reasonably stable. 

## Installation

Currently the easiest way to install a development version is to clone this repo locally, change to the folder, and use `poetry`:

	poetry install -v 
	poetry run pytest


## Contributing

Contributions are welcomed, either in the form of raising issues or pull requests on this repo. Please take a look at the Code of Conduct before interacting, which includes instructions for reporting any violations.

## Coming soon(ish)

- PyPI package
- Standard optimization strategies (PCSeg and correlation-consistent)

## Documentation

For dependencies, detailed installation instructions, and a guide to getting started, please refer to the main documentation [here](https://basisopt.readthedocs.io/en/latest/index.html).

## Examples

There are working examples in the examples folder, and these are (or will be) documented in the documentation. 

## Acknowledging usage

If you use this library in your program and find it helpful, that's great! Any feedback would be much appreciated. If you publish results using this library, please consider citing the following paper detailing the implementation:

[ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/640f48e3b5d5dbe9e832e997)
