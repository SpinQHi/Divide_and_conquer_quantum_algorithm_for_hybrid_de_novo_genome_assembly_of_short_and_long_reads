import multiprocessing
from Computing_loss_for_6_nodes_all_Ansatz import train_dvqe_nparts as dvqe
from Hamiltonian_transformation import prepare_hamiltonian
import numpy as np



def main(seed):
    
    loss, path, converge_Itr = dvqe(
            hamiltonian_list=hamiltonian_list, 
            num_qubit=qubits_num, 
            num_part=num_part,
            cir_depth=cir_depth, 
            learning_rate=learning_rate, 
            itr_num=itr_num, 
            seed=seed, 
            ansatz=ansatz,
            ground_energy=ground_state_energy
        )
            
            
if __name__ == '__main__':    
    
    num_process = 8
    itr_num = 500
    cir_depth = 2

    learning_rate= 0.2
    ansatz_pool = ['HEA1','HEA2','Propose_Circuit']
    ground_state_energy = -30
    qubits_num = 60
    nodes = 6
    num_part = int(qubits_num / nodes)
    
    hamiltonian_path = f'Hamiltonian/{qubits_num}_qubits_hamiltonian.txt'
    hamiltonian_list = prepare_hamiltonian(hamiltonian_path,qubits_num)
    
    for ansatz in ansatz_pool:
        # Initialize the seed pool
        seed_pool = np.load(f'Data_Pool/Searched_Seed_Pool/{ansatz}_Searched_Seed_Pool.npy')
        with multiprocessing.Pool(num_process) as p:
            arr = p.map(main,  (seed for seed in seed_pool))
