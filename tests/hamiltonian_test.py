import itertools
import numpy as np
from src.cross_hamiltonian import SpinBosonHL, Jaynes_Cummings_Hamiltonian as JCM
from qiskit.quantum_info import SparsePauliOp, Statevector
from qutip import *

rng = np.random.default_rng()

def jaynes_cummings_ham(Delta: int = 0, g: int = 1, cutoff = 2):
    return JCM(Delta, g, cutoff)

   
def test_full_hamiltonian():
    print("Testing full JCM Hamiltonian...")
    Delta, g, cutoff = 0, 1, 4
    h = jaynes_cummings_ham(Delta=Delta/2, g=g, cutoff=cutoff)
    # h_full = 0.5*Delta * tensor(sigmaz(), identity(cutoff)) \
    #    + g * (tensor(sigmap(), destroy(cutoff)) + tensor(sigmam(), create(cutoff)))

    h_full = 0.5 * Delta * tensor(identity(cutoff), sigmaz()) \
            + g * (tensor(destroy(cutoff), sigmap()) + tensor(create(cutoff), sigmam()))
    
    print(h.full_hamiltonian().data.toarray())
    print("\n\n")
    print(h_full.data.toarray())
    assert h_full == h.full_hamiltonian()
    print("test_full_hamilotnian() test passed\n")


def test_map_to_qubits():
    print("Testing map_to_qubits...")
    h = SpinBosonHL([([sigmaz()], 1.0)], "test1", [1], 0)
    qubit_op = SparsePauliOp.from_list([('Z', 1.0)])
    assert qubit_op.equiv(h.map_to_qubits())

    delta, g, cutoff = 1, 1, 2
    h = jaynes_cummings_ham(Delta=delta, g=g, cutoff=cutoff)
    qubit_op = SparsePauliOp.from_list([('ZI', 0.5*delta), ('XX', 0.5*g), ('YY', -0.5*g)])

    print(qubit_op)
    print(h.map_to_qubits())

    assert qubit_op.equiv(h.map_to_qubits())

    h = SpinBosonHL([([create(4)], 1.0)], "test2", [0], 4)
    assert Statevector.from_int(0, 4).evolve(h.map_to_qubits()).equiv(Statevector.from_int(1, 4))

    h = SpinBosonHL([([create(4), sigmaz()], 1.0)], "test3", [0], 4)
    ket_a = Statevector.from_int(0, 2).tensor(Statevector.from_int(0, 4))
    ket_b = Statevector.from_int(0, 2).tensor(Statevector.from_int(1, 4))
    ket_c = Statevector.from_int(1, 2).tensor(Statevector.from_int(0, 4))
    ket_d = Statevector.from_int(1, 2).tensor(Statevector.from_int(1, 4))

    assert ket_b.equiv(ket_a.evolve(h.map_to_qubits()))
    assert Statevector(-1.0*ket_d.data).equiv(ket_c.evolve(h.map_to_qubits()))
    print("test_map_to_qubits() test passed\n")
