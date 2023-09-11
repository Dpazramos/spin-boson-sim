import itertools
import numpy as np
from src.hamiltonian import *
from qiskit.quantum_info import SparsePauliOp
from qutip import *

rng = np.random.default_rng()

def jaynes_cummings_ham(Delta: int = 0, g: int = 1, cutoff = 2):

    h_tls = ([sigmaz(), identity(cutoff)], 0.5 * Delta)
    h_int1 = ([sigmap(), destroy(cutoff)], g)
    h_int2 = ([sigmam(), create(cutoff)], g)

    photon_ind = [1]

    H_list = [h_tls, h_int1, h_int2]

    return HamiltonianList(H_list, photon_ind, cutoff)

def test_full_hamiltonian():
    Delta, g, cutoff = 0, 1, 4
    h = jaynes_cummings_ham(Delta=Delta, g=g, cutoff=cutoff)
    h_full = 0.5*Delta * tensor(sigmaz(), identity(cutoff)) \
        + g * (tensor(sigmap(), destroy(cutoff)) + tensor(sigmam(), create(cutoff)))
    assert h_full == h.full_hamiltonian()

def test_map_to_qubits():
    h = HamiltonianList([([sigmaz()], 1.0)], [1], 0)
    qubit_op = SparsePauliOp.from_list([('Z', 1.0)])
    assert qubit_op.equiv(h.map_to_qubits())

    delta, g, cutoff = 1, 1, 2
    h = jaynes_cummings_ham(Delta=delta, g=g, cutoff=cutoff)
    qubit_op = SparsePauliOp.from_list([('IZ', 0.5*delta), ('XX', 0.5*g), ('YY', -0.5*g)])
    assert qubit_op.equiv(h.map_to_qubits())

    h = HamiltonianList([([create(4)], 1.0)], [0], 4)
    assert Statevector.from_int(0, 4).evolve(h.map_to_qubits()).equiv(Statevector.from_int(1, 4))

    h = HamiltonianList([([create(4), sigmaz()], 1.0)], [0], 4)
    ket_a = Statevector.from_int(0, 2).tensor(Statevector.from_int(0, 4))
    ket_b = Statevector.from_int(0, 2).tensor(Statevector.from_int(1, 4))
    ket_c = Statevector.from_int(1, 2).tensor(Statevector.from_int(0, 4))
    ket_d = Statevector.from_int(1, 2).tensor(Statevector.from_int(1, 4))

    assert ket_b.equiv(ket_a.evolve(h.map_to_qubits()))
    assert Statevector(-1.0*ket_d.data).equiv(ket_c.evolve(h.map_to_qubits()))
