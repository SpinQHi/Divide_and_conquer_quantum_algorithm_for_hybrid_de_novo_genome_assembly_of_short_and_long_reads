def spilt_hamiltonian_nparts(hamiltonian_list, num_qubit, num_part):
    """
    Split the Hamiltonian obtained from 'prepare_hamiltonian' into 
    coef list and subsystems for A, B, C Hamiltonians.
    """
    # Create lists to store coefficients and complete Hamiltonian in string format
    coefs_list = []
    pauli_str_list = []

    for i in range(len(hamiltonian_list)):
        coefs_list.append(hamiltonian_list[i][1])
        pauli_str_list.append(hamiltonian_list[i][0])

    # Number of qubits for the first 'num_parts - 1' subsystems
    sub_qubit = num_qubit // num_part

    # List to store Hamiltonian Pauli strings for all subsystems
    sub_list = []

    # Create first 'num_parts - 1' subsystems
    for k in range(num_part - 1):
        pauli_sub_list = []
        
        for j in range(len(pauli_str_list)):        
            pauli_sub_list.append((pauli_str_list[j][k * sub_qubit: (k+1) * sub_qubit], 1))
        
        # Store the subsystem
        sub_list.append(pauli_sub_list)
    
    # Create the last subsystem
    pauli_sub_list = []
    
    for j in range(len(pauli_str_list)):        
        pauli_sub_list.append((pauli_str_list[j][(num_part - 1) * sub_qubit:], 1))     

    # Store the subsystem
    sub_list.append(pauli_sub_list)           

    # Return lists
    return coefs_list, sub_list
