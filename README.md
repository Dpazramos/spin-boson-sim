# Extended Spin-boson simulation
Tools for digital quantum simulation of spin-boson Hamiltonians in Qiskit backends. Extended from https://github.com/ahaukis/spin-boson-sim to include broader OOP-oriented design, Compatibility with SLURM queue computing, and greater flexibility in the implementation of physical systems for simulation (with respect to the Hamiltonian and initial state).

## Contents
-`src/cross_hamiltonian.py`: Classes for defining Hamiltonians with QuTiP, Qiskit and Numpy representations.
- `src/cross_operator.py`: Classes for defining quantum operators with QuTiP, Qiskit and Numpy representations.
-`src/cross_state`: Class for defining quantum states with QuTiP, Qiskit and Numpy representations.
- `src/sim.py`: Classes for defining and executing simulations, using exact QuTiP simulations, as well as simulated quantum circuits using Qiskit, for Trotterization, Zero Noise Extrapolation, and ISL [1]
- `src/inline_sim.py`: Run TCM or HSC systems with pre-set parameters in the code.
- `src/args_sim.py`: Run TCM or HSC systems with parsed arguments passed in to the code.
- `src/args_hs.py`: Parses HSC-specific arguments to run a simulation.
- `src/args_tc.py`: Parses TCM-specific arguments to run a simulation.
- `arrjob.sh`: SLURM submission script used for running simulations based on input parameters.
- `src/plot/plotter.py`: Contains a basic procedure for plotting the data obtained from a simulation, for different methods.
- `src/utils/`: Contains some utility functions for data parsing, visualising, and constants
- `src/tests`:  Contains some tests for QuTiP and `SparsePauliOp` conversion, as well as Hamitonian class construction.

Not ehtat the code used in the "cross" classes for converting QuTiP objects to `SparsePauliOp`s  (Qiskit), is as described in Ref. [2]

[1] Jaderberg, B., Agarwal, A., Leonhardt, K. et al. Minimum hardware requirements for hybrid quantum–classical DMFT. Quantum Sci. Technol. 5, 034015 (2020). https://doi.org/10.1088/2058-9565/ab972b

[2] Sawaya, N.P.D., Menke, T., Kyaw, T.H. et al. Resource-efficient digital quantum simulation of d-level systems for photonic, vibrational, and spin-s Hamiltonians. npj Quantum Inf 6, 49 (2020). https://doi.org/10.1038/s41534-020-0278-0

## Requirements
- numpy
- qiskit~=0.44.1
- qutip
- pytest
- mitiq~=0.30.0
- qiskit-experiments~=0.5.4
- qiskit-aer~=0.12
- For the `tc.py` script, the ISL part requires a local installation of the `qiskit-0441` branch of https://github.com/ahaukis/isl

Remember to install Jupyter, to run src/tests/run_test.ipynb.