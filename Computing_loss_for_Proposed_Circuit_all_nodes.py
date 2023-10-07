
import re
import numpy as np
import time
import torch
from matplotlib import pyplot as plt

from spinqkit.algorithm import VQE
from spinqkit import get_basic_simulator, get_compiler, BasicSimulatorConfig, generate_hamiltonian_matrix
from spinqkit.algorithm.optimizer.torch_optim import TorchOptimizer
from spinqkit.backend.pytorch_backend import TorchSimulator
from spinqkit.algorithm.Expval import _scipy_sparse_mat_to_torch_sparse_tensor
from spinqkit import generate_hamiltonian_matrix, draw
from spinqkit.algorithm.optimizer.gradientdescent import GradientDescent
from spinqkit.model.parameter import Parameter

from Ansatz import build_HEA1_circuit, build_HEA2_circuit, build_proposed_circuit
from Spilt_hamiltonian import spilt_hamiltonian_nparts


def measurement_result(measurement_str, num_qubit):
    """
    Pad zeros to the input binary string according to the number of qubits.
    """
    return measurement_str.zfill(num_qubit)



class DistributedVQE_nparts(torch.nn.Module):
    """
    The class for implementing distributed Variational Quantum Eigensolver (VQE) 
    with the Hamiltonian divided into 'n' parts.
    """
    def __init__(self, num_qubit, num_part, cir_depth, hamiltonian_list, seed,ansatz):
        super(DistributedVQE_nparts, self).__init__()
        
        torch.manual_seed(seed)
        np.random.seed(seed=seed)  
        
        self.dtype = torch.complex64
        self.device = torch.device('cpu')
        self.cir_depth = cir_depth
        self.num_part = num_part  
        self.num_qubit = num_qubit 
        
        coefs_list, sub_list = spilt_hamiltonian_nparts(hamiltonian_list, num_qubit, num_part)
        self.coefs_list = torch.tensor(coefs_list).reshape([-1, 1, 1])
        self.sub_list = sub_list

        sub_hamiltonian = []
        circuit_param = []
        exe_list = []
        compiler = get_compiler('native')
        for i in range(num_part):
          
            sub_hamiltonian.append([_scipy_sparse_mat_to_torch_sparse_tensor(generate_hamiltonian_matrix([ham]))
                                        for ham in sub_list[i]])
            
            if ansatz == 'HEA1':
                ansatz_params = Parameter(np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * len(sub_list[i][0][0]) * cir_depth),
                              trainable=True)
                circuit = build_HEA1_circuit(qubit_num=len(sub_list[i][0][0]), circuit_depth=cir_depth,ansatz_params = ansatz_params)
            elif ansatz == 'HEA2':
                ansatz_params = Parameter(np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * len(sub_list[i][0][0]) * cir_depth),
                              trainable=True)
                circuit = build_HEA2_circuit(qubit_num=len(sub_list[i][0][0]), circuit_depth=cir_depth,ansatz_params = ansatz_params)
            else:
                ansatz_params = Parameter(np.random.uniform(0.3 * np.pi, 0.5 * np.pi, 2 * len(sub_list[i][0][0])),
                              trainable=True)
                circuit = build_proposed_circuit(qubit_num=len(sub_list[i][0][0]),ansatz_params = ansatz_params)
            circuit_param.append(circuit.params)
                
                
                
            circuit_param.append(circuit.params)
        
            exe_list.append(compiler.compile(circuit, 0))
            # draw(compiler.compile(circuit, 0),filename='q.png')

        self.parameter_num_each_qubit = int(len(circuit_param[0])/(self.cir_depth * len(sub_list[0][0][0])))
  
        self.sub_hamiltonian = sub_hamiltonian    
        self.params = torch.nn.Parameter(torch.from_numpy(np.concatenate(circuit_param)),
                                   requires_grad=True)

        self.exe = exe_list
        self.sim = TorchSimulator()        
 
            
    def forward(self):
        
        state_list = []
        sub_expval = torch.tensor(1.0)
        sub_qubit = self.num_qubit // self.num_part

        for i in range(self.num_part - 1):
            init_states = torch.eye(1, 2 ** (len(self.sub_list[i][0][0])), ).to(self.device, self.dtype)
           
            final_states = torch.permute(self.sim._get_final_state(self.exe[i], init_states,
                                                               self.params[self.parameter_num_each_qubit*i*self.cir_depth*sub_qubit:self.parameter_num_each_qubit*(i+1)*self.cir_depth*sub_qubit],
                                                               self.exe[i].qnum), dims=[1, 0])
            k_value = torch.concat([torch.unsqueeze(ham @ final_states, dim=0) for ham in self.sub_hamiltonian[i]])
            expval_value = torch.real(final_states.conj().T @ k_value)
            
            sub_expval = sub_expval * expval_value
            state_list.append(final_states)

        
        init_states = torch.eye(1, 2 ** (len(self.sub_list[-1][0][0])), ).to(self.device, self.dtype)
        final_states = torch.permute(self.sim._get_final_state(self.exe[-1], init_states,
                                                            self.params[self.parameter_num_each_qubit*(self.num_part - 1)*self.cir_depth*sub_qubit:],
                                                            self.exe[-1].qnum), dims=[1, 0])
        k_value = torch.concat([torch.unsqueeze(ham @ final_states, dim=0) for ham in self.sub_hamiltonian[-1]])
        expval_value = torch.real(final_states.conj().T @ k_value)
        
        sub_expval = sub_expval * expval_value
        state_list.append(final_states)
        
    

        matrix = torch.sum(self.coefs_list * sub_expval, dim=0)
        eigval = matrix[0][0]
        
        return eigval, state_list


def train_dvqe_nparts(hamiltonian_list, num_qubit, num_part,cir_depth, learning_rate, itr_num, seed, ansatz,ground_energy):

    start_time = time.time()   
    np.random.seed(seed=seed)

    dvqe = DistributedVQE_nparts(num_qubit=num_qubit, num_part=num_part, cir_depth=cir_depth, 
                                 hamiltonian_list=hamiltonian_list, seed=seed,ansatz=ansatz)

    optimizer = torch.optim.Adam(dvqe.parameters(), lr=learning_rate)

    summary_loss = [0]  

    for itr in range(itr_num):
        optimizer.zero_grad()
     
        loss, state_list = dvqe()
    
        loss.backward()
      
        optimizer.step()
    
    
        if (itr+1) % 1 == 0:
            print(f"iter: {itr+1}, loss: {loss: .8f}, time: {time.time()-start_time:.2f}s")
            
        if np.abs(loss.detach().numpy() - summary_loss[-1]) < 1e-12:  
            print(f'The loss difference less than {1e-12}. Optimize done')
            summary_loss.append(loss.detach().numpy())
            break 
        
        if loss < ground_energy + 0.2:
            print(f'The seed is found and the seed is {seed} while the loss is {loss}\n')
            break
        summary_loss.append(loss.detach().numpy())
       
    _, sub_list = spilt_hamiltonian_nparts(hamiltonian_list, num_qubit, num_part)
    measure_sting = ''
    
    for i in range(num_part):

        final_state_idx = torch.max(torch.abs(state_list[i]),dim=0)[1]

        result = str(bin(np.array(final_state_idx)[0])[2:])

        measure_sting += measurement_result(result, len(sub_list[i][0][0]))

    return summary_loss[1:], measure_sting, itr    

