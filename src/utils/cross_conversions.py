from qiskit.quantum_info import SparsePauliOp
from qutip import Qobj


def graycode(n: int) -> str:
    """Returns a binary string representing integer n in the Gray code."""
    gray = n^(n>>1)
    return bin(gray)[2:]

def singlequbit_ketbra(i, j) -> SparsePauliOp:
    """Converts |i><j| = |0><0|, |0><1|, |1><0| or |1><1| to the corresponding
    single-qubit Pauli operators.
    
    Args:
        i (int): Index of ket-vector |i>.
        j (int): Index of bra-vector <j|.

    Returns:
        SparsePauliOp
    """

    match (i,j):
        case (0,0):
            # |0><0| = (I + Z)/2
            return SparsePauliOp.from_list([('I', 0.5), ('Z', 0.5)])
        case (0,1):
            # |0><1| = (X + iY)/2
            return SparsePauliOp.from_list([('X', 0.5), ('Y', 0.5j)])
        case (1,0):
            # |1><0| = (X - iY)/2
            return SparsePauliOp.from_list([('X', 0.5), ('Y', -0.5j)])
        case (1,1):
            # |1><1| = (I - Z)/2
            return SparsePauliOp.from_list([('I', 0.5), ('Z', -0.5)])
        case _:
            raise ValueError("Single-qubit operator indices must be 0 or 1.")


def ketbra(n: int, k: int, n_qubits: int, encoding: str = 'standard') -> SparsePauliOp:
    """Convert operator |n><k| into the corresponding Pauli operator.

    Args:
        n (int): Basis state index of ket-vector |n>.
        k (int): Basis state index of bra-vector <k|.
        n_qubits (int): Number of qubits.
        encoding (str): How n and k are converted to binary strings.
            Options: standard (binary) and graycode.

    Returns:
        SparsePauliOp
    """
    if n > 2**n_qubits-1 or k > 2**n_qubits-1:
        raise ValueError("Operator index too high for qubit number.")

    match encoding:
        case "standard":
            n, k = bin(n)[2:], bin(k)[2:]
        case "graycode":
            n, k = graycode(n), graycode(k)
        case _:
            raise ValueError(f"Invalid encoding: {encoding}")

    # pad to full length
    n = n.rjust(n_qubits, '0')
    k = k.rjust(n_qubits, '0')

    op = singlequbit_ketbra(int(n[0]), int(k[0]))
    for i in range(1, n_qubits):
        op = op.tensor( singlequbit_ketbra(int(n[i]), int(k[i])) )

    return op

def qobj_to_sparsepauliop(qobj: Qobj, n_qubits: int, encoding: str = 'standard') -> SparsePauliOp:
    """
    Convert a (bosonic) Qobj into a SparsePauliOp.
    The returned operator assumes Qiskit qubit ordering.
    """
    mat = qobj.data.tocoo()
    # n_qubits = int(np.log2(mat.shape[0]))
    # print("n_qubits:", n_qubits, "log:", int(np.log2(mat.shape[0])))

    # empty object
    if mat.data.size == 0:
        return SparsePauliOp(''.join(['I' for _ in range(n_qubits)]), 0.0)
    elif mat.shape == (2, 2):   # more efficient for a single-qubit operator
        a, b, c, d = qobj.data[0,0], qobj.data[0,1], qobj.data[1,0], qobj.data[1,1]
        return SparsePauliOp.from_list([('I', (a+d)*0.5), ('X', (b+c)*0.5), ('Y', (b-c)*0.5j), ('Z', (a-d)*0.5)])
    
    op = mat.data[0] * ketbra(mat.row[0], mat.col[0], n_qubits, encoding)
    for i, j, coeff in zip(mat.row[1:], mat.col[1:], mat.data[1:]):
        op += coeff * ketbra(i, j, n_qubits, encoding)

    return op.simplify()
