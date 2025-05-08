# Utility functions for proper qubit index tracking
from qiskit import QuantumCircuit
from qiskit.transpiler.layout import TranspileLayout

# Get the final layout, if any, determined by the TranspileLayout object
def get_last_layout(layout):
    if layout is None:
        return None
    else:
        final_layout = layout.final_layout

        if final_layout is None:
            init_layout = layout.initial_layout
            return init_layout
        else:
            return final_layout

# Get the layout of the virtual/logical qubits of the input layout 
def get_layout_qubits(layout, n_logical_qubits: int):
    last_layout = get_last_layout(layout)

    if last_layout is None:
        physical_qubits  = {i: i for i in range(n_logical_qubits)}
        return physical_qubits
    else:
        virtual_to_physical = {virtual: physical for physical, virtual in last_layout.get_physical_bits().items()}

        # print("virtual to physical:\n", virtual_to_physical)

        # Get the physical qubits from the virtual_to_physical dictionary
        physical_qubits = {}
        for qubit, phys_ind in virtual_to_physical.items():
            qubit_register = qubit.register

            if "ancilla" in qubit_register.name.lower():
                is_ancilla = True
            else:
                is_ancilla = False

            if not is_ancilla:
                physical_qubits[qubit.index] = phys_ind
            
        physical_qubits = {qubit_ind: virtual_ind for qubit_ind, virtual_ind in physical_qubits.items() if qubit_ind < n_logical_qubits}
        return physical_qubits 

# Construct an 'initial_layout' list
def get_initial_qubits(layout, n_logical, n_physical):
    init_virtual = get_layout_qubits(layout, n_logical)
    return mapping_to_initial_qubits(init_virtual, n_logical, n_physical)


# Convert the logical qubit mapping into a list encoding the mapping (list[x] = index to which the xth logical qubit is mapped to)
def mapping_to_initial_qubits(physical_qubits: dict, n_logical: int, n_physical: int):
    if len(physical_qubits) == 0 or (physical_qubits is None):
        return list(range(n_physical))
    
    init_vq = [physical_qubits[q] for q in range(n_logical)]
    rest = [q for q in range(n_physical) if q not in init_vq]
    initial_qubits = init_vq + rest
    return initial_qubits

# Remove the qubits that are not used in the circuit
def truncate_to_active_qubits(circuit, virtual_layout = None):
    # Track active qubits
    active_qubits = set()
    for instr, qargs, _ in circuit.data:
        for q in qargs:
            active_qubits.add(q)

    # Convert active qubits to a sorted list for deterministic ordering
    active_qubit_list = sorted(active_qubits, key=lambda q: circuit.qubits.index(q))
    # print("Active qubits list:", active_qubit_list)
    active_indices = {q: idx for idx, q in enumerate(active_qubit_list)}
    
    # Handle reordering based on virtual layout
    last_layout = get_last_layout(virtual_layout)

    if last_layout:
        last_layout_virtual = last_layout.get_virtual_bits()
        # print("virtual:", last_layout_virtual)

        # Map active qubits to virtual layout by matching indices
        reorder_map = {
            q: last_layout_virtual[next(v for v in last_layout_virtual if v.index == q.index)]
            for q in active_qubit_list
            if any(v.index == q.index for v in last_layout_virtual)
        }

        # Check for missing mappings
        if len(reorder_map) != len(active_qubit_list):
            missing_qubits = [q for q in active_qubit_list if all(v.index != q.index for v in last_layout_virtual)]
            raise ValueError(f"Some active qubits are missing in the virtual layout: {missing_qubits}")
   
        # print("Reorder map:", reorder_map)
        sorted_active_qubits = sorted(active_qubit_list, key=lambda q: reorder_map[q])
        # print("Sorted active qubits:", sorted_active_qubits)
        active_indices = {q: idx for idx, q in enumerate(sorted_active_qubits)}
    else:
        sorted_active_qubits = active_qubit_list

    # Create a new circuit with the correct number of qubits
    truncated_circuit = QuantumCircuit(len(active_indices), circuit.num_clbits)
    
    # Rebuild the circuit data for the truncated circuit
    for instr, qargs, cargs in circuit.data:
        if all(q in active_indices for q in qargs):
            new_qargs = [truncated_circuit.qubits[active_indices[q]] for q in qargs]
            truncated_circuit.append(instr, new_qargs, cargs)
    
    return truncated_circuit


# Update the coupling map after transpilation has been performed
def update_coupling_map(original_coupling_map, active_indices):
    # Filter and remap the coupling map
    updated_coupling_map = []
    for pair in original_coupling_map:
        if pair[0] in active_indices and pair[1] in active_indices:
            updated_coupling_map.append([active_indices[pair[0]], active_indices[pair[1]]])

    return updated_coupling_map
