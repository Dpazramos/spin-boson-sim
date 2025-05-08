# Define quantum operators, with compatible representation as Qobj() instances (QuTiP) and Quantum Circuits (Qiskit)
# Can be used interchangeably as observables

from qiskit_aer.backends.compatibility import DensityMatrix
from qiskit.quantum_info import (Operator, SparsePauliOp)
from qiskit.transpiler.layout import TranspileLayout
from qiskit.result.result import Result
from qiskit.primitives import BackendEstimator
from qiskit import AncillaRegister
from qiskit import QuantumCircuit, execute
import qutip as qt
import numpy as np

from abc import (ABC, abstractmethod)
from typing import (Union, Tuple)

from utils.qindex import get_layout_qubits, mapping_to_initial_qubits

# Define the abstract cross-operator class - defined for any unitary operator, 
###############################
### Abstract Cross-Operator ###
###############################
class CrossOperator(ABC):
    """
    Abstract class for defining quantum states as both Qiskit Operatorss and QuTiP Qobjs, with added functionality for simulations
    """
    # Qiskit Operator representation of the state
    @property
    def qk_op(self) -> Operator:
        Operator(self.data)

    # Convert the operator to a Qiskit circuit
    @property
    def qk_circ(self) -> QuantumCircuit:
        inst = self.qk_op.to_instruction()
        qc = QuantumCircuit()
        return qc.from_instructions([inst])
    
    # Number of qubits in the state
    @property
    def n_qubits(self) -> int:
        return np.log2(self.data.shape[0])

    # Operator as a Qobj
    @property
    def qt_op(self) -> qt.Qobj:
        return qt.Qobj(self.mirror_data)

    # Operator as an nd-array in the computational basis
    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        pass

    # Operator mirrored for QuTiP ordering
    @property
    @abstractmethod
    def mirror_data(self) -> np.ndarray:
        pass

    # String identification of the state
    @property
    @abstractmethod
    def name(self) -> str:
        pass


########################
### Cross-Observable ###
########################
class CrossObservable(CrossOperator):
    """
    Added functionality for expectation values, for observables with real eigenvalues
    """
    # QuTiP expectation value
    def qt_expect(self, state: qt.Qobj) -> float:
        return qt.expect(self.qt_op, state)
    
    # Compute with Qiskit
    def qk_expect(self, results, _) -> float:
            
        if isinstance(results, Result):
            rho = results.data()["density_matrix"]
            data = self.data

            if isinstance(rho, DensityMatrix):
                rho = rho.data
            else:
                raise TypeError(f"Invalid results type: {type(rho)}")
            
            return np.real(np.trace(rho@data))
            
        else:
            print(f"ERROR - Invalid results type ({type(results)}).")
            return None

    def run_qc(self, target_qc: QuantumCircuit, backend, shots: int = 1, run_estimator: bool = True, optimization_level: int = 0, n: int = None, init_layout = None) -> dict:
        qc = target_qc.copy()
        results = {}

        # Extract the layouts
        target_layout = target_qc.layout
        physical_qubits_from_target = get_layout_qubits(target_layout, n) if target_layout else {}
        physical_qubits_from_init = get_layout_qubits(init_layout, n) if init_layout else {}

        # Merge layouts (init_layout takes precedence over target_layout)
        physical_qubits = {**physical_qubits_from_target, **physical_qubits_from_init}
    
        if len(physical_qubits) == 0:
            physical_qubits = {i: i for i in range(n)}

        # print("Physical qubits (merged):\n", physical_qubits)

        # Generate the combined initial qubits list (includes ancillas)
        initial_qubits = mapping_to_initial_qubits(physical_qubits, n, len(qc.qubits))

        # print("initial layout:\n", init_layout)
        # print("target layout:\n", target_layout)
        # print("physical qubits (merged):\n", physical_qubits)
        # print("initial qubits (list):\n", initial_qubits)
        # print("len(qc.qubits)", len(qc.qubits))

        # If run_estimator is False, execute the circuit
        if not run_estimator:
            qc.measure_all()
            qc.save_density_matrix()

            # Execute the circuit
            job = execute(
                qc,
                backend,
                optimization_level=optimization_level,
                basis_gates=backend.configuration().basis_gates + ['save_density_matrix'],
                shots=shots,
                initial_layout=initial_qubits,
            )

            result = job.result()

            counts = result.get_counts()
            # print("counts:", counts)
            # print("physical qubits:", physical_qubits)

            # Map counts to logical qubits
            mapped_counts = {}
            for bit_str, count in counts.items():
                reversed_bit_str = bit_str[::-1]

                # Create a new logical bitstring using physical_qubits
                logical_bitstring = ['0'] * n
                for logical_index, physical_index in physical_qubits.items():
                    logical_bitstring[logical_index] = reversed_bit_str[physical_index]

                logical_bitstring = ''.join(logical_bitstring)
                mapped_counts[logical_bitstring] = mapped_counts.get(logical_bitstring, 0) + count

            val = self.qk_expect(result, mapped_counts)
        else:
            # Use BackendEstimator if run_estimator is True
            estimator = BackendEstimator(backend)
            operator = SparsePauliOp.from_operator(self.data)
            job = estimator.run(qc, operator, shots=shots)
            result = job.result()
            val = result.values[0]

        results["result"] = result
        results["value"] = val

        return results

    # Observable data type
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    # Flag whether the observable is a manual observable
    @property
    @abstractmethod
    def manual(self) -> bool:
        pass

    # *** Observable range *** #
    @property
    @abstractmethod
    def l_bound(self) -> float:
        pass

    @property
    @abstractmethod
    def u_bound(self) -> float:
        pass


#########################
### Input Cross-State ###
#########################
class InputObservable(CrossObservable):
    """
    State defined by an user-input state and bit string (and an optional basis change circuit)
    """
    def __init__(self, name: str, data: np.ndarray, mirror_data: np.ndarray, type = None, lbound = None, ubound = None, manual = False):
        self._data = data
        self._name = name
        self._mirror_data = mirror_data
        self._type = type if type is not None else "Input" 
        self._lbound = lbound if lbound is not None else -np.inf
        self._ubound = ubound if ubound is not None else np.inf
        self._manual = manual

    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def mirror_data(self) -> np.ndarray:
        return self._mirror_data

    @property
    def name(self):
        return self._name
    
    @property
    def type(self):
        return self._type

    @property
    def l_bound(self):
        return self._lbound
    
    @property
    def u_bound(self):
        return self._ubound
    

    @property
    def manual(self):
        return self._manual

    def to_dict(self):
        return {
            "_type": "InputObservable",
            "name": self.name,
            "data": self.data,
            "mirror_data": self.mirror_data,
            "type": self.type,
            "lbound": self.l_bound,
            "ubound": self.u_bound,
        }

    @staticmethod
    def from_dict(d):
        name = d["name"]
        state_data = InputObservable.data_from_dict(d["data"])
        mirror_data = InputObservable.data_from_dict(d["mirror_data"])
        type = d.get("type", None)
        lbound, ubound = d.get("lbound", None), d.get("ubound", None)
        return InputObservable(name, state_data, mirror_data, type=type, lbound=lbound, ubound=ubound)
    

    @staticmethod
    def data_from_dict(d_obj):
        in_data = np.array(d_obj)
        out_data = np.zeros_like(in_data)

        for i, row in enumerate(in_data):
            for j, val in enumerate(row):
                out_data[i, j] = complex(val["real"], val["imag"])
        
        return out_data

##############################
### Bit-string Cross-State ###
##############################
class BitObservable(InputObservable):
    """
    Observable corresponding to the probability of measuring the given bit string
    """
    def __init__(self, bitstring: str = None, manual_counts = False, keep_counts=[]):
        assert all(bit in ['0', '1'] for bit in bitstring), "Invalid bitstring."
        # Qiskit-order data (little-endian)
        rev_bitstring = bitstring[::-1]
        data = Operator.from_label(rev_bitstring).data

        # QuTiP order data (big-endian)
        mirror_data = Operator.from_label(bitstring).data

        self.keep = keep_counts

        if len(keep_counts) > 0:
            data_type = "mitigated_probabilities"
            manual_counts = True

        elif manual_counts:
            data_type = "manual_probabilities"

        else:
            data_type = "probabilities"

        super().__init__(bitstring, data, mirror_data, data_type, lbound=0, ubound=1,manual=manual_counts)

        if manual_counts:
            self.qk_expect = self._counts_qk_expect


    # Overwrite the expectation value - instead compute the average of times the bitstring is found.
    def _counts_qk_expect(self, _, mapped_counts : dict) -> float:
        # bit_str = self.name[::-1]   # Target bitstring (Accounting for Qiskit ordering)
        bit_str = self.name # No need to account for Qiskit ordering, as this is done in mapped_counts
        bit_counts = mapped_counts.get(bit_str, 0)

        # print("-------------- mapped counts:\n", mapped_counts, "\n--------------")
        # print("bitstring:", bit_str)

        # Compute total shots based on `self.keep` if defined
        if len(self.keep) == 0:
            # Sum over all counts if `self.keep` is not specified
            shots = sum(mapped_counts.values())
        else:
            # Only sum counts for bitstrings in `self.keep`
            shots = sum(count for bit, count in mapped_counts.items() if bit in self.keep)
            
        # Avoid division by zero
        if shots == 0:
            print("ERROR - Total shots are zero.")
            return None

        # Return the frequency of the target bitstring
        return bit_counts / shots


    # Convert to dictionary to allow deserialization (for JSON)
    def to_dict(self):
        return {
            "_type": "BitObservable",
            "bitstring": self.name,
            }
    
    @staticmethod
    def from_dict(d):
        bitstring = d["bitstring"]
        bit_state = BitObservable(bitstring)
        return bit_state


class RepeatedBitObs(BitObservable):
    """
    Observable corresponding to a bitstring of equal value to '1'
    """
    def __init__(self, n_qubits : int = None, type : str = None, manual_counts = False, keep_counts=[]):
        # Match the type
        if type == "ones" or type == "1":
            bitstring = '1'*n_qubits
        elif type == "zeros" or type == "0":
            bitstring = '0'*n_qubits
        else:
            raise ValueError(f"Invalid name for repeated bitstring observable ({type}). Must be 'ones' or 'zeros'.")
        
        super().__init__(bitstring=bitstring, manual_counts=manual_counts, keep_counts=keep_counts)


########################
### Spin Observables ###
########################
class SpinObservableSingle(InputObservable):
    """
    Observable corresponding to a single Pauli operator acting on a single qubit
    """
    def __init__(self, ind: Union[int, str] = None, qubit_ind: int = None, n_qubits: int = None):
        # Shorthand for Pauli operators
        pauli = self.pauli_ops

        if isinstance(ind, int):
            assert(pauli_ind in range(4)), f"Invalid Pauli index.\n(given {ind}, must be in [0, 1, 2, 3] (or correspondingly, in ['I', 'X', 'Y', 'Z'])"
            pauli_keys = list(pauli.keys())
            pauli_ind = pauli_keys[ind]
            spin = pauli[pauli_ind]
        else:
            ind = ind.upper()
            assert(ind in ['I', 'X', 'Y', 'Z']), f"Invalid Pauli index.\n(given {pauli_ind}, must be in ['I', 'X', 'Y', 'Z'] (or correspondingly, in [0, 1, 2, 3])"
            pauli_ind = ind
            spin = pauli[pauli_ind]

        data = 1
        rev_data = 1

        for i in range(n_qubits):
            if i == qubit_ind:
                data = np.kron(data, spin)          # little-endian
                rev_data = np.kron(spin, rev_data)  # big-endian
            else:
                data = np.kron(data, pauli["I"]) # Identity operator (little-endian)
                rev_data = np.kron(pauli["I"], rev_data) # Identity operator (big-endian)


        super().__init__(f"{pauli_ind}_({qubit_ind},{n_qubits})", data, rev_data, type="spin", lbound=-1, ubound=1)


    @property
    def pauli_ops(self) -> dict:
        # Define the Pauli axes - and corresponding (single-qubit) operators
        pauli_ops = {
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
            "I": np.eye(2),
        }
        return pauli_ops


# Canonical axis Spin observable
class AddedSpinObservable(InputObservable):
    """
    Observable corresponding to the added value of spins in a given axis, for a system
    """
    def __init__(self, axes: list = None):
        # Axes is a List containing the corresponding axis to measure for each qubit (e.g. ['x', 'x', 'y', 'x', 'z', 'y'] for a 6-qubit system)
        assert set(axes).issubset(['X', 'Y', 'Z', 'I', '0']), "Invalid axis. Must be in ['X', 'Y', 'Z', 'I', '0']"

        n_qubits = len(axes)

        # Construct the string name
        name = axes.copy()
        # name.reverse()
        name = '+'.join(name)

        data = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
        rev_data = np.zeros_like(data)

        # axes.reverse() # Invert for little-endian ordering
        for i, ax in enumerate(axes):
            if ax == "0":
                continue
            else:
                sub_op = SpinObservableSingle(ax, i, n_qubits)
                data += sub_op.data
                rev_data += sub_op.mirror_data

        super().__init__(name, data, rev_data, type="spin", lbound=-n_qubits, ubound=n_qubits)


# Repeated spin observable
class RepeatedAddSpinObservable(AddedSpinObservable):
    """
    Spin Observable with the same axis for all qubits
    """

    def __init__(self, n_qubits: int = None, axis: str = None):
        axis_cap = axis.capitalize()
        assert axis_cap in ['X', 'Y', 'Z', 'I'], "Invalid axis. Must be in ['X', 'Y', 'Z', 'I']"
        axes = [axis_cap for _ in range(n_qubits)]
        super().__init__(axes)
