from src.hamiltonian import *
#coding: utf-8
#!/usr/bin/env python

# Example command:
# python -u "C:\Users\Daniel Paz\OneDrive\Escritorio\Aalto\MQS\code\alisa_thesis\spin-boson-sim\tc.py" 1 0.01 20 1024 0.01 -o output --noisy_isl
# Runs the simulation for N=1 atom, with delta_t=0.1, 20 steps, 1024 shots, 1e-2 tolerance, storing the results in the output folder, and considering noisy_isl simulations

import argparse
from isl.recompilers import ISLRecompiler, ISLConfig
import numpy as np
import os
import pickle
import qutip as qt

from qiskit import execute, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeNairobi
from qiskit_aer import AerSimulator

parser = argparse.ArgumentParser()
parser.add_argument("n_atoms", help="number of two-level systems", type=int)
parser.add_argument("delta_t", help="size of time step", type=float)
parser.add_argument("n_steps", help="number of time steps", type=int)
parser.add_argument("shots", help="shots per circuit evaluation", type=int)
parser.add_argument("tolerance", help="tolerance during ISL", type=float)
parser.add_argument("-o", "--output", help="name of output folder", default="output")
parser.add_argument("--noisy_isl", action="store_true", help="use noise backend during ISL")

args = parser.parse_args()

output_folder = args.output
try:
    os.mkdir(output_folder)
except FileExistsError:
    pass

#############################
### SIMULATION PARAMETERS ###
#############################

n_atoms = args.n_atoms # two-level systems, 1 qubit per each
n_photon_qubits = 1  # qubits per photonic mode
n_qubits = n_atoms + n_photon_qubits  # in total
cutoff = 2**n_photon_qubits

# TCM params
omega = 1
g = 10

# Trotterized simulation parameters
delta_t = args.delta_t
steps = args.n_steps
T = delta_t * steps
t = np.linspace(0, T, steps+1)

h = tavis_cummings_ham(n_atoms, cutoff, omega, g)
h_qt = h.full_hamiltonian()
h_q = h.map_to_qubits('graycode')   # quantum_info class

trotter = PauliEvolutionGate(h_q, time=t[1])

#######################
### EXACT EVOLUTION ###
#######################

psi0 = qt.tensor(*([qt.basis(cutoff, 1)] + [qt.basis(2, 1) for _ in range(n_atoms)])).unit()
exact_t = np.linspace(0, T, 5*steps+1)
exact_result = qt.sesolve(h_qt, psi0, exact_t, [])
qt.qsave(exact_result, output_folder+"/qutip_data")

exact_expect = []
for state in exact_result.states:
    exact_expect.append(np.abs(state.overlap(psi0))**2)
np.savetxt(output_folder+"/qutip_probabilities.out", exact_expect)


###################
### TROTTERIZED ###
###################

backend = AerSimulator.from_backend(FakeNairobi())
shots = args.shots
circuit_list = []
trotter_results = []

print("Trotter start.")
for i in range(t.size):
    circ = QuantumCircuit(n_qubits)
    circ.x(range(n_qubits)) # initial state: |11...1>
    for j in range(i):
        circ.append(trotter, circ.qubits)
    circuit_list.append(circ.copy())
    circ.save_density_matrix()
    circ.measure_all()
    result = execute(circ, backend=backend, shots=shots).result()
    trotter_results.append(result)
    print("{}/{}".format(i+1, t.size))
print("Trotter end.\n")
with open(output_folder+"/trotter_results.pickle", 'wb') as f:
    pickle.dump(trotter_results, f, pickle.HIGHEST_PROTOCOL)

trotter_probabilities = []
for i in range(t.size):
    trotter_probabilities.append(trotter_results[i].get_counts().get("1"*n_qubits, 0)/shots)
np.savetxt(output_folder+"/trotter_probabilities.out", trotter_probabilities)

with open(output_folder+"/trotter_circuits.pickle", 'wb') as f:
    pickle.dump(circuit_list, f, pickle.HIGHEST_PROTOCOL)


###########
### ISL ###
###########

# num of CNOTs may not exceed largest trotter circuit
max_circuit = transpile(circuit_list[-1], backend)
max_layers = max_circuit.depth()
n_2q_gates = max_circuit.count_ops()['cx']

config = ISLConfig(max_layers=max_layers,
                   sufficient_cost=args.tolerance,
                   max_2q_gates=n_2q_gates,
                   method='ISL')

state_prep_circ = QuantumCircuit(n_qubits)
state_prep_circ.x(range(n_qubits))

isl_results = []
isl_circuits = []
previous_circ = state_prep_circ

print("ISL start.")
for i in range(t.size):
    target_circ = previous_circ
    if i>0:
        target_circ.append(trotter, target_circ.qubits)
    recompiler_kwargs = {'isl_config':config,
                         'coupling_map':backend.configuration().coupling_map}
    if args.noisy_isl:
        recompiler_kwargs['backend'] = backend
    recompiler = ISLRecompiler(target_circ, execute_kwargs={'shots':shots}, **recompiler_kwargs)
    result = recompiler.recompile()
    isl_results.append(result)
    previous_circ = result['circuit'].copy()
    isl_circuits.append(result['circuit'])
    print("step {}/{}\t{:.4f}\t{:.2f} min".format(i+1, t.size, result['overlap'], result['time_taken']/60))
print("ISL end.\n")
with open(output_folder+"/isl_results.pickle", 'wb') as f:
    pickle.dump(isl_results, f, pickle.HIGHEST_PROTOCOL)

isl_qiskit_results = []
isl_probabilities = []
for i in range(t.size):
    circ = isl_circuits[i].copy()
    circ.save_density_matrix()
    circ.measure_all()
    result = execute(circ, backend, shots=shots).result()
    isl_qiskit_results.append(result)
    isl_probabilities.append(result.get_counts().get("1" * n_qubits, 0) / shots)
with open(output_folder+"/isl_qiskit_results.pickle", 'wb') as f:
    pickle.dump(isl_qiskit_results, f, pickle.HIGHEST_PROTOCOL)
np.savetxt(output_folder+"/isl_probabilities.out", isl_probabilities)
