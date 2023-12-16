# Spin-boson simulation

Tools for simulating spin-boson Hamiltonians with Qiskit.

## Contents

- `src/hamiltonian.py`: Class for defining Hamiltonians with QuTiP objects.
- `src/conversions.py`: Convert (bosonic) QuTiP objects to `SparsePauliOp`s (Qiskit) as described in Ref. [1].
- `tc.py`: A script for simulating the time-evolution of a Tavis-Cummings system with QuTiP, Trotterization via Qiskit, and ISL.

[1] Sawaya, N.P.D., Menke, T., Kyaw, T.H. et al. Resource-efficient digital quantum simulation of d-level systems for photonic, vibrational, and spin-s Hamiltonians. npj Quantum Inf 6, 49 (2020). https://doi.org/10.1038/s41534-020-0278-0

## Requirements

- numpy
- qiskit~=0.44.1
- qutip
- pytest
- For the `tc.py` script, a local installation of the `qiskit-0441` branch of https://github.com/ahaukis/isl