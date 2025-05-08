# Define quantum states, with compatible representation as Qobj() instances (QuTiP) and Quantum Circuits (Qiskit)
# Can be used interchangeably as observables

from qiskit.quantum_info import Statevector, Operator
from qiskit import QuantumCircuit
import qutip as qt

from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np

# Define the abstract cross-state class
############################
### Abstract Cross-State ###
############################
class CrossState(ABC):
    """
    Abstract class for defining quantum states to use for state evolution and observables
    """
    # Compute the probability given an input state (qutip/ndarray)
    def inner_prod(self, state: Union[qt.Qobj, np.ndarray]) -> float:
        if isinstance(state, qt.Qobj):
            return state.overlap(self.qt_state)
        else:
            return np.vdot(state, self.data)
        

    # Get the state as a Qobj
    @property
    def qt_state(self) -> qt.Qobj:
        return qt.Qobj(self.mirror_data)
        # return qt.Qobj(self.data)
    

    # State as a Qiskit circuit
    @property
    def as_qk_circuit(self) -> QuantumCircuit:
        inst = self.qk_op.to_instruction()
        qc = QuantumCircuit()
        return qc.from_instructions([inst])


    # Get the number of qubits
    @property
    def n_qubits(self) -> int:
        return np.log2(self.data.shape[0])


    # Get the state as a StateVector
    @property
    def qk_state(self) -> Statevector:
        return Statevector(self.data)


    # Define the corresponding projection operator
    @property
    def qk_proj(self) -> Operator:
        return self.qk_state.to_operator()


    # Define the corresponding operator required for turning the |0> state into the cross_state
    @property
    @abstractmethod
    def qk_op(self) -> Operator:
       pass

    # State as an nd-array in the computational basis
    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        pass


    # String identification of the state
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    
    # Mirror state data, to match QuTiP ordering
    @property
    @abstractmethod
    def mirror_data(self) -> np.ndarray:
        pass
    

#########################
### Input Cross-State ###
#########################
class InputState(CrossState):
    """
    State defined by an user-input state and bit string (and an optional basis change circuit)
    """
    def __init__(self, name: str, data: np.ndarray, mirror_data: np.ndarray, qk_op: Union[QuantumCircuit, Operator]):
        self._data = data
        self._name = name
        self._qk_op = qk_op
        self._mirror_data = mirror_data


    @property
    def qk_op(self) -> QuantumCircuit:
        if isinstance(self._qk_op, QuantumCircuit):
            qk_op = Operator.from_circuit(self._qk_op)
            return qk_op
        elif isinstance(self._qk_op, Operator):
            return qk_op
        else:
            raise ValueError(f"Invalid operator type ({type(qk_op)})")


    @property
    def name(self):
        return self._name
    

    @property
    def data(self) -> np.ndarray:
        return self._data
    
    
    @property
    def mirror_data(self) -> np.ndarray:
        return self._mirror_data


    def to_dict(self):
        return {
            "_type": "InputState",
            "name": self._name,
            "data": self._data.tolist(),
            "mirror_data": self._mirror_data.tolist(),
            "qk_circ": self.as_qk_circuit.qasm(),
        }

    @staticmethod
    def from_dict(d):
        name = d["name"]
        state_data = np.array(d["data"])
        mirror_data = np.array(d["mirror_data"])
        qk_circ = QuantumCircuit.from_qasm_str(["qk_circ"])
        return InputState(name, state_data, mirror_data, qk_circ)


##############################
### Bit-string Cross-State ###
##############################
class BitState(InputState):
    """
    State defined by computational basis bit string
    """
    def __init__(self, bitstring: str = None):
        assert all(bit in ['0', '1'] for bit in bitstring), "Invalid bitstring."
        self._n_qubits = len(bitstring)

        # Get the bitstring in reverse order for proper Qiskit ordering
        rev_bitstring = bitstring[::-1]
        qobj = qt.tensor(*[qt.basis(2, int(bitstring[i])) for i in range(self._n_qubits)]).unit()
        qobj_rev = qt.tensor(*[qt.basis(2, int(rev_bitstring[i])) for i in range(self._n_qubits)]).unit()

        self._name = bitstring # Set name before initiating to avoid circular definitions

        super().__init__(bitstring, qobj.full(), qobj_rev.full(), self.as_qk_circuit)


    # Overwrite for a simple circuit (TODO: Verify this)
    @property
    def as_qk_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self._n_qubits)
        bit_list = self.name[::-1]
        for i, bit in enumerate(bit_list):
            if bit == '1':
                circuit.x(i)        # NOTE: Qiskit uses reverse bit ordering
        return circuit

    # Convert to dictionary to allow deserialization (for JSON)
    def to_dict(self):
        return {"_type": "BitState",
                "bitstring": self.name,
            }
    
    @staticmethod
    def from_dict(d):
        bitstring = d["bitstring"]
        bit_state = BitState(bitstring)
        return bit_state


class RepeatedBitState(BitState):
    """
    Observable corresponding to a bitstring of equal value to '1'
    """
    def __init__(self, n_qubits : int = None, type : str = None):
    
        if type == 'ones':
            bitstring = '1'*n_qubits
        elif type == 'zeros':
            bitstring = '0'*n_qubits
        else:
            raise ValueError(f"Invalid name for repeated bitstring observable ({type}). Must be 'ones' or 'zeros'.")
            
        super().__init__(bitstring)


# Superposition bitstring states

class PsiRest(InputState):
    """
    State defined by $|\Psi\rangle \otimes |1\rangle^{\otimes n - 2}$, where n > 1 is the number of qubits 
    
    'rest' is a list of ones and zeros, indicating the state of the remaining qubits (as eigenstates of the Z operator)
    """
    def __init__(self, rest: list = []):
        self._n_qubits = 2 + len(rest)
        name = ','.join([str(bit) for bit in rest])
        self._name = f"Psi{name}_{self._n_qubits}"
        self._rest = rest

        # Define the basis states
        zero = [1, 0]
        one = [0, 1]

        # Define the state
        state = PsiRest.psi()
        revstate = PsiRest.psi()

        if len(rest) > 0:
            for qubit in rest:
                assert qubit in [0, 1, '0', '1'], "Invalid qubit value."

                add_state = zero if str(qubit) == '0' else one
                state = np.kron(state, add_state)
                revstate = np.kron(add_state, revstate)
        

        super().__init__(self._name, state, revstate, self.as_qk_circuit)


    @staticmethod
    def psi(plus=True):
        r_phase = 1 if plus else -1
        psi = [0, 1, r_phase * 1, 0]
        return np.array(psi) / np.linalg.norm(psi)
    

    @property
    def as_qk_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self._n_qubits)

        rev_rest = self._rest[::-1]

        for i, bit in enumerate(rev_rest):
            if bit == 1:
                circuit.x(i)

        circuit.h(self._n_qubits - 2)
        circuit.cx(self._n_qubits - 1, self._n_qubits - 2)
    
        return circuit
    

    
# class SuperpositionBitstringsFirst(InputState):
#     """
#     State defined by a uniform superposition of two bitstrings, corresponding to one excitation
    
#     Entangled inds is a list of tuples, where the first element refers to the index (big-endian) of the entangled qubit, 
#     and the second element is a boolean indicating whether the entanglement pattern 
#     (i.e. upon an ideal measurement, matching with all other qubits in the list with a same value.)
    
#     """
#     def __init__(self, entangled_inds = list[Tuple[int, bool]],  n_qubits: int = None, rest_bits = 'ones'):
#         if len(entangled_inds) == 0:
#             entangled_inds = [[0, True], [1, True]]

#         self.entangled_inds = entangled_inds

#         self._name = f"Supos_{entangled_inds}_{rest_bits}"

#         self._n_qubits = n_qubits
#         self._name = f"suppos_{entangled_inds}_{n_qubits}"

#         self.data = "?"

#         self.mirror_data = "¿"

#         super().__init__(self._name, self.data, self.mirror_data, self.as_qk_circuit)


#     @property
#     def as_qk_circuit(self) -> QuantumCircuit:
#         circuit = QuantumCircuit(self._n_qubits)
#         entangled_list = self.entangled_inds[::-1] # NOTE: Qiskit uses reverse bit ordering
#         for i, (bit, value) in enumerate(entangled_list):
#             if i == 0:
#                 circuit.h(bit)
#             else:
#                 circuit.cx(entangled_list[0], bit[0])

#                 if not value:
#                     circuit.x(bit)    

#         return circuit

#     # Auxiliary function to get matrix representation of the state
#     @staticmethod
#     def to_data(entangled_bits, n_qubits):
#         data = np.zeros(2**n_qubits)

#         entangled_inds = [entangled_data[0] for entangled_data in entangled_bits]
#         entangled_values = [entangled_data[1] for entangled_data in entangled_bits]

#         return 0
    