import re

def read_txt(path1):
    file1 = open(path1, 'r')
    string = file1.read()
    return string

def helper(H):
    H = H.replace('-', '+-').split('+')
    res = []
    for word in H:
        digit = ''
        pauli = ''
        site = False
        if word:
            for c in word:
                if c in {'-', '.'} or c.isnumeric():
                    if not site:
                        digit += c
                    else:
                        pauli += c
                elif c.isalpha():
                    if site:
                        pauli += ','
                    else:
                        site = True
                    pauli += c.upper()
            res.append([float(digit), pauli if pauli else 'I'])
    return res

def prepare_hamiltonian(path_in, num_qubits):
    hamiltonian = helper(read_txt(path_in))
    new_hamiltonian = list()
    for idx, (coeff, pauli_term) in enumerate(hamiltonian):
        pauli_term = re.split(r',\s*', pauli_term.upper())
        pauli_list = ['I'] * num_qubits
        for item in pauli_term:
            if len(item) > 1:
                pauli_list[int(item[1:])] = item[0]
            elif item[0].lower() != 'i':
                raise ValueError('Expecting I for ', item[0])
        new_term = (''.join(pauli_list), coeff)
        new_hamiltonian.append(new_term)
    return new_hamiltonian