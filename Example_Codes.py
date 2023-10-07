import multiprocessing
from Computing_Loss import train_dvqe_nparts as dvqe
from Hamiltonian_transformation import prepare_hamiltonian
import numpy as np
import torch


def main(seed):
    for ansatz in ansatz_pool:
        for index, qubits_num in enumerate(qubit_num_pool):
            num_part = int(qubits_num / nodes_pool[index])
            print(f'num_part = {num_part}')
            hamiltonian_path = f'Hamiltonian/{qubits_num}_qubits_hamiltonian.txt'
            hamiltonian_list = prepare_hamiltonian(hamiltonian_path,qubits_num)
            ground_energy = ground_state_energy_pool[index]
            
            loss, path, converge_Itr = dvqe(
                hamiltonian_list=hamiltonian_list, 
                num_qubit=qubits_num, 
                num_part=num_part,
                cir_depth=cir_depth, 
                learning_rate=learning_rate, 
                itr_num=itr_num, 
                seed=seed, 
                ansatz=ansatz,
                ground_energy=ground_energy
            )
            
            if loss[-1] < ground_energy + 0.5:
                np.save(f'Result\{nodes_pool[index]}\Loss\{qubits_num}_qubits_model_loss_{seed}.npy',loss)
                np.save(f'Result\{nodes_pool[index]}\Converge_ITR\{qubits_num}_qubits_model_converge_Itr_{seed}.npy',converge_Itr)   
            
if __name__ == '__main__':    
    # Initialize the seed pool
    seed_pool = []

    # Set the target size for the seed pool
    target_size = 1000

    # Generate seeds until the seed pool reaches the target size
    while len(seed_pool) < target_size:
        # Randomly generate an integer between 1 and 1024 as a seed
        seed = np.random.randint(1,8192)

        # If the generated seed is not in the seed pool, add it to the seed pool
        if seed not in seed_pool:
            seed_pool.append(seed)
    
    num_process = 30
    itr_num = 500
    cir_depth = 2
    ground_state_energy = 10
    learning_rate= 0.2
    ansatz_pool = ['HEA1','HEA2','Propose_Circuit']
    ground_state_energy_pool = [-30,-35,-40,-45,-49]
    qubit_num_pool = [60,77,96,117,140]
    nodes_pool = [6,7,8,9,10]
    
    for qubits_num in qubit_num_pool:
        with multiprocessing.Pool(num_process) as p:
            arr = p.map(main,  (seed for seed in seed_pool))
    