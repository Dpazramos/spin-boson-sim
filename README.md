## Fork information
This fork has been adapted to further expand upon the work carried out in Alisa Haukisalmi's master thesis

(https://aaltodoc.aalto.fi/items/d3900ebc-d9cf-4cde-a4f6-01e84a31c1d1)

---

# Spin-boson simulation

Tools for simulating spin-boson Hamiltonians with Qiskit.

## Contents

- `src/hamiltonian.py`: Class for defining Hamiltonians with QuTiP objects.
- `src/conversions.py`: Convert (bosonic) QuTiP objects to `SparsePauliOp`s (Qiskit) as described in Ref. [1].
- `tc.py`: A script for simulating the time-evolution of a Tavis-Cummings system with QuTiP, Trotterization via Qiskit, and ISL [2].

[1] Sawaya, N.P.D., Menke, T., Kyaw, T.H. et al. Resource-efficient digital quantum simulation of d-level systems for photonic, vibrational, and spin-s Hamiltonians. npj Quantum Inf 6, 49 (2020). https://doi.org/10.1038/s41534-020-0278-0

[2] Jaderberg, B., Agarwal, A., Leonhardt, K. et al. Minimum hardware requirements for hybrid quantum–classical DMFT. Quantum Sci. Technol. 5, 034015 (2020). https://doi.org/10.1088/2058-9565/ab972b

## Requirements

- numpy
- qiskit~=0.44.1
- qutip
- pytest
- For the `tc.py` script, the ISL part requires a local installation of the `qiskit-0441` branch of https://github.com/ahaukis/isl
