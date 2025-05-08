# ***** Define constants for use in various functions *****
from cross_operator import (BitObservable, RepeatedBitObs, 
                            SpinObservableSingle, RepeatedAddSpinObservable, 
                            AddedSpinObservable)
from cross_hamiltonian import (CrossHamiltonian, 
                               Tavis_Cummings_Hamiltonian, Jaynes_Cummings_Hamiltonian, 
                               Heisenberg_Hamiltonian)
from cross_state import (InputState, BitState, RepeatedBitState, PsiRest)

from qiskit.circuit.library import (SXGate, IGate, XGate, YGate, ZGate, CXGate)

# Define the folder input names, and result output base names
results_io = {
    "Exact_trotter_results": "Exact_trotter",
    "Noisy_trotter_results": "Noisy_trotter",
    "Exact_isl_qiskit_results": "Exact_isl",
    "Noisy_isl_qiskit_results": "Noisy_isl",
    "Noisy_isl_1e-2_qiskit_results": "Noisy_isl_1e-2",
    "Noisy_isl_1e-4_qiskit_results": "Noisy_isl_1e-4",
    "qutip_results": "qutip",
}


# Define the plot styles for the different types of data
plot_styles = {
    "qutip": ("Exact", ":", "black"),
    "Exact_isl": ("Exact ISL", "--", "pink"),
    "Noisy_isl": ("Noisy ISL", None, "red"),
    "Exact_trotter": ("Exact Trotterized", "--", "cyan"),
    "Noisy_trotter": ("Noisy Trotterized", None, "blue"),
    "zne": ("ZNE", None, "purple"),
    "isl": ("Noisy ISL", None, "red"),
}

# Define the dictionaries for Cross-Observable/State types
observable_dict = {
    "bitstring": BitObservable,
    "repbit": RepeatedBitObs,
    "spin": SpinObservableSingle,
    "addspin": AddedSpinObservable,
    "repaddspin": RepeatedAddSpinObservable,
    "TC": Tavis_Cummings_Hamiltonian,
    "JC": Jaynes_Cummings_Hamiltonian,
    "HS": Heisenberg_Hamiltonian,
}

state_dict = {
    "bitstring": BitState, 
    "repbit": RepeatedBitState,
    "psirest": PsiRest,
}

# Define the JSON keys for the different types of data
json_keys = {
    "BitState": BitState,
    "InputState": InputState,
    "CrossHamiltonian": CrossHamiltonian,
}

# Define the valid bitstring permutations for the Tavis-Cummings model with one excitation
tc_perms = [["00", "11"], ["001", "111", "010"], ["1111", "0011", "0101", "0110"]]

# Define the name aliases for single qubit gates
obs_alias = {
    'id': IGate(), 
    'sx': SXGate(), 
    'x': XGate(), 
    'y': YGate(), 
    'z': ZGate(),
    'cx': CXGate(),
}
