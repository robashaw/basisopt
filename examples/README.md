# Examples
These scripts demonstrate examples of BasisOpt usage in common basis set optimisation tasks. This includes:

- Multi-molecule uses the Psi4 program to re-optimise the def2-SVP basis set for a set of 5 molecules (the molecule xyz files are included). The optimisation uses the wB97X-D density functional approximation and loops until a pre-defined incremental change between iterations is reached. This example also demonstrates how to use the logging functionality to print user-defined messages.

- Reduce takes the large Partridge Uncontracted 3 primitives from the Basis Set Exchange and systematically removes the most redundant exponents until a 10s5p configuration is achieved. The Psi4 package is used.

Note that several of these example scripts set a temporary directory for Psi4. This should be modified to an appropriate directory on your system.


