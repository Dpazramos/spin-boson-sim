import pytest
from src.utils.cross_conversions import *
from qutip import *
from qiskit.quantum_info import SparsePauliOp, Statevector

def test_singlequbit_ketbra():
    assert singlequbit_ketbra(0, 0).equiv(SparsePauliOp.from_list([('I', 0.5), ('Z', 0.5)]))
    assert singlequbit_ketbra(1, 1).equiv(SparsePauliOp.from_list([('I', 0.5), ('Z', -0.5)]))
    assert singlequbit_ketbra(1, 0).equiv(SparsePauliOp.from_list([('X', 0.5), ('Y', -0.5j)]))
    assert singlequbit_ketbra(0, 1).equiv(SparsePauliOp.from_list([('X', 0.5), ('Y', 0.5j)]))

    with pytest.raises(ValueError):
        singlequbit_ketbra(0, 2)

def test_ketbra():
    assert ketbra(0, 0, 1, 'standard').equiv(singlequbit_ketbra(0, 0))

    # |1><0| on two qubits
    op = SparsePauliOp.from_list([('IX', 0.25), ('IY', -0.25j), ('ZX', 0.25), ('ZY', -0.25j)])
    assert op.equiv(ketbra(1, 0, 2, encoding='standard'))

    ket_0 = Statevector.from_int(0, 2**2)
    ket_1 = Statevector.from_int(1, 2**2)
    assert ket_0.evolve(ketbra(1, 0, 2, encoding='standard')).equiv(ket_1)
    assert ket_1.evolve(ketbra(1, 0, 2, encoding='standard')).equiv(Statevector([0,0,0,0]))

    # |0><4| in std encoding
    op = SparsePauliOp.from_list([('XII', 0.125), ('YII', 0.125j), \
                                 ('XZI', 0.125), ('XIZ', 0.125), ('YZI', 0.125j), ('YIZ', 0.125j), \
                                 ('XZZ', 0.125), ('YZZ', 0.125j)])
    assert op.equiv(ketbra(0, 4, 3, 'standard'))
    assert Statevector.from_int(4, 2**3).evolve(ketbra(0, 4, 3, 'standard')).equiv(Statevector.from_int(0, 2**3))

    with pytest.raises(ValueError):
        ketbra(0, 4, n_qubits=2)

def test_qobj_to_sparsepauliop():
    # n ~= 0|0><0| + 1|1><1| = |1><1|
    n = num(2)
    n_bin = qobj_to_sparsepauliop(n, 1, encoding='standard')
    n_gray = qobj_to_sparsepauliop(n, 1, encoding='graycode')
    assert n_bin.equiv(SparsePauliOp.from_list([('I', 0.5), ('Z', -0.5)]))
    assert n_bin.equiv(n_gray)

    # n ~= 0|0><0| + 1|1><1| + 2|2><2| = |1><1| + 2|2><2|
    n = num(3)
    print(n)
    n_bin = qobj_to_sparsepauliop(n, 2, encoding='standard')
    print(n_bin)
    n_gray = qobj_to_sparsepauliop(n, 2, encoding='graycode')
    assert n_bin.equiv(SparsePauliOp.from_list([('II', 0.75), ('ZI', -0.25), ('IZ', 0.25), ('ZZ', -0.75)]))
    assert n_gray.equiv(SparsePauliOp.from_list([('II', 0.75), ('ZI', -0.25), ('IZ', -0.75), ('ZZ', 0.25)]))

    # a ~= sqrt(1)|0><1| = |0><1|
    a = destroy(2)
    a_bin = qobj_to_sparsepauliop(a, 1, 'standard')
    assert a_bin.equiv(SparsePauliOp.from_list([('X', 0.5), ('Y', 0.5j)]))

    with pytest.raises(ValueError):
        a = destroy(3)
        qobj_to_sparsepauliop(a, 1, 'asdfghj')
