'''
This file contains utility classes and functions to be used in the main algorithm, which are tested to be bug-free.
However, they are still subject to potential efficiency improvements in the future, as these utility classes/functions
are used over again in the main algorithm.
'''

import numpy as np
import random

# The QuantumState class, where each instance of this class represents a properly normalized quantum state
# Characterized by its instance variables "state" (type=[complex]), "separable" (type=bool), "n_qubits" (type=int), "components" (type=[QuantumState])
class QuantumState:

    # initializes the instance, defining the state according to a given init_state or random generation
    # Hereby makes sure every QuantumState instance represents a properly normalized state
    def __init__(self, init_state=None, n_qubits=0):
        if init_state:      # if a initial state is designated
            self.state = init_state
            self.n_qubits = int(np.log2(len(init_state)))
        elif (not init_state) and (n_qubits > 0):               # for random generation
            state = []
            for _ in range(2 ** n_qubits):
                state.append(complex(random.randint(-10, 10), random.randint(-10, 10)))
            state = QuantumUtils.normalize(state)
            self.state = state
            self.n_qubits = n_qubits
        else:
            raise Exception("Error 0: quantum state illegally initialized.")
        
        self.components = []
        self.separable = False


    def set_separability(self, sep):
        self.separable = sep
    

    def add_component(self, comp):
        self.components.append(comp)
    

    def multiply(self, another_state):      # returns the multiplication state of self and another QuantumState instance, with self on the MSB side
        state1 = self.state
        state2 = another_state.state
        psi = []
        for i in state1:
            for j in state2:
                psi.append(i * j)

        product = QuantumState(init_state=psi)
        product.set_separability(True)
        if not self.components:
            product.add_component(self)
        if not another_state.components:
            product.add_component(another_state)

        for comp in self.components:
            product.add_component(comp)
        for comps in another_state.components:
            product.add_component(comps)

        return product
    

    # generate the MxN A matrix for a particular combination (k, n-k) (only these k numbers are needed), for state psi
    # basis states of amplitude 0 are ignored
    def get_A(self, comb):      
        n = self.n_qubits
        psi = self.state
        result = [[], [], []]       # result[0] is a list of basis states, of which amplitudes are nonzero, for k qubits
                                    # result[1] is that for n-k qubits
                                    # result[2] is the A matrix generated, A[i][j] is the amplitude for basis result[0][j] result[1][i]
        A = []
        A_tool = 0              # record the positions of k and n-k qubits
        B_tool = 0
        for a in comb:
            A_tool += 2 ** a
        B_tool = 2 ** n - 1 - A_tool

        for basis in range(len(psi)):      
            basis_A = basis & A_tool             # basis state corresponding to these k qubits
            basis_B = basis & B_tool             # basis state corresponding to these n-k qubits
                                                # Here basis_A and basis_B will not overlap in binary

            if psi[basis] != 0.0:                       # record these basis states in result
                # print("basis: ", basis)
                if basis_A not in result[0]:
                    result[0].append(basis_A)
                if basis_B not in result[1]:
                    result[1].append(basis_B)
        result[0].sort()
        result[1].sort()

        # Now result[0] and result[1] are finished
        # Now we create the A matrix with max spatial efficiency
        for b in result[1]:
            A.append([])
            for a in result[0]:
                basis = a | b     # construct the total basis state from two sets of qubits
                A[len(A)-1].append(psi[basis])
        result[2] = np.matrix(A, dtype=complex)
        return result






# The QuantumUtils class, as a collection of utility functions for the algorithm
class QuantumUtils:

    # returns a normalized statevector given the statev (type=[complex]).
    # If mode == 0, do the proper normalization (length = 1)
    # elif mode == 1, do the expansive normalization (normalize the abs of the smallest entry to 1)
    def normalize(statev, mode=0):
        normalizer = 0.0
        if mode == 0:
            for amp in statev:
                normalizer += abs(amp) ** 2
            normalizer = np.sqrt(normalizer)            
        else:
            normalizer = min(statev)
        
        result = list(np.array(statev) / normalizer)
        return result


    # Generate a list of all combinations of k items from the list [0, 1, 2, 3, ..., n-1], only produce a half if k = n / 2
    def generate_combs(n, k):
        def recu_combs(L, k):   # recursively generate all combinations of k items from the list L
            if len(L) < k:
                return []
            elif len(L) == k:
                return [L]
            elif k == 1:
                temp = []
                for i in L:
                    temp.append([i])
                return temp
            else:
                listA = recu_combs(L[1:], k - 1)       # combinations containing L[0]
                for i in listA:
                    i.append(L[0])
                listB = recu_combs(L[1:], k)    # combinations not containing L[0]
                return listA + listB

        n_list = [i for i in range(n)]
        first_result = recu_combs(n_list, k)
        if k == n / 2.0:    # remove a half of combinations
            second_result = []
            for a in first_result:
                exists_complement = False
                for b in second_result:
                    if sorted(a + b) == n_list:         # if its complement list is already recorded in second_result
                        exists_complement = True
                if not exists_complement:
                    second_result.append(a)
            return second_result
        else:
            return first_result
    
    
    # rounds a complex matrix M up to n decimal places
    def round_matrix(M, n):
        N = M.shape
        for i in range(N[0]):
            for j in range(N[1]):
                M[i, j] = complex(round(M[i, j].real, n), round(M[i, j].imag, n))
        return M

    
    # compare elements of two rows of the same dimensions, up to n decimal places, with a tolerance fraction
    def compare_row(a, b, n, tol=0.6):        
        same = True
        a = QuantumUtils.round_matrix(a, n)
        b = QuantumUtils.round_matrix(b, n)
        for i in range(a.size):
            if abs(a[0, i] - b[0, i]) > min(tol * abs(b[0, i]), tol * abs(a[0, i])):
                same = False
        return same


    # Randomly generate a separable psi consisting of n_states multiplied together, for n qubits
    def generate_separable_psi(n, n_states):
        n_qubits = []   # number of qubits of each component state
        states = []     # the list of all component states, order is from LSB to MSB
        n_qubit_upper = n - n_states + 1        # initial upper limit of number of qubits available to a new component state
                                                # such that every other state has at least one qubit
        for i in range(n_states):   # for every new component state
            new_state = []
            if i == n_states - 1:       # if we are dealing with the last state
                n_qubit = n_qubit_upper
            else:
                n_qubit = random.randint(1, n_qubit_upper)
                n_qubit_upper -= n_qubit
                n_qubit_upper += 1
            n_qubits.append(n_qubit)

            new_state = QuantumState(n_qubits=n_qubit)
            states.append(new_state)

        # print(n_qubits)
        # Now we generate psi from these states
        result = states[0]   
        for i in range(1, len(states) - 1):
            result = result.multiply(states[i])

        
        return result, states
    
    