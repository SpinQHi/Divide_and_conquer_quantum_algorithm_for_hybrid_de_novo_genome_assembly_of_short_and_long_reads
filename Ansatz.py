from spinqkit.model import Circuit, Rx, Ry, Rz, H, CZ, X, CX
from spinqkit.algorithm.Expval import ExpvalCost

import numpy as np

def build_HEA1_circuit(qubit_num, circuit_depth,ansatz_params):
    """
    Build a Hardware-Efficient Ansatz (HEA) type 1 circuit.
    Ry + Rz + CZ
    
    Args:
        qubit_num (int): Number of qubits.
        circuit_depth (int): Depth of the circuit.

    Returns:
        Circuit: The constructed HEA circuit.
    """

    circuit = Circuit(ansatz_params)
    circuit.allocateQubits(qubit_num)

    for d in range(circuit_depth):
        for q in range(qubit_num):
            circuit << (Ry, [q], lambda x, idx=2*qubit_num*d + 2*q: x[idx])
            circuit << (Rz, [q], lambda x, idx=2*qubit_num*d + 2*q + 1: x[idx])

        if qubit_num > 1:
            for q in range(qubit_num - 1):
                circuit.append(CZ, [q, q + 1])

    return circuit


def build_HEA2_circuit(qubit_num, circuit_depth,ansatz_params):
    """
    Build a Hardware-Efficient Ansatz (HEA) type 2 circuit.   
    Only Ry + Rz
    
    Args:
        qubit_num (int): Number of qubits.
        circuit_depth (int): Depth of the circuit.

    Returns:
        Circuit: The constructed HEA circuit.
    """

    
    circuit = Circuit(ansatz_params)
    circuit.allocateQubits(qubit_num)

    for d in range(circuit_depth):
        for q in range(qubit_num):
            circuit << (Ry, [q], lambda x, idx=2*qubit_num*d + 2*q: x[idx])
            circuit << (Rz, [q], lambda x, idx=2*qubit_num*d + 2*q + 1: x[idx])

    return circuit


def build_proposed_circuit(qubit_num,ansatz_params):
    """
    Build the proposed circuit.

    Args:
        qubit_num (int): Number of qubits.
        circuit_depth (int): Depth of the circuit. Defaults to 1.

    Returns:
        Circuit: The constructed circuit.
    """
    
    circuit = Circuit(ansatz_params)
    circuit.allocateQubits(qubit_num)

    circuit << (Rx, [0], lambda x, idx=0: x[idx])

    if qubit_num > 1:
        for q in range(qubit_num - 1):
            circuit << (Ry, [q+1], lambda x, idx=q+1: x[idx])
            circuit.append(CZ, [q, q+1])
            circuit << (Ry, [q+1], lambda x, idx=q+1: x[idx] + np.pi)
        
        for qx in range(qubit_num - 1, 0, -1):
            circuit.append(CX, [qx-1, qx])

    return circuit
