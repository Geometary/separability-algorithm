''' This algorithm is based on the idea of Schmidt decomposition, where an arbitrary state vector in n-qubit Hilbert space is 
    decomposed into a sum of products of Schmidt coefficients and Schmidt
    vectors.

    Note that this algorithm only tests for FULL separability (i.e. whether
    the initial state can be broken into a SINGLE product of two sums, and
    so on). Also, only the performance of the MAIN algorithm is tested (i.e. performances of subsidiary parts, 
    say generation of A matrix, are ignored.) Besides, qubits are in ascending order from RIGHT to LEFT.

    INPUT:
        - n, total number of qubits, integer.
        - psi, the initial state vector of dimension 2^n (length n), in which each
           component represents the probability amplitude in that
           basis state.

    OUTPUT:
        - separable, a boolean variable entailing if psi is separable.
        - states, a list of vectors (or equivalently, a matrix) that multiply
           together to produce psi, and are not further reducible.'''

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from algo_utils import *

print_f = open('record.txt', 'w+')


def normalize(psi):     # normalize the state psi
    normalizer = 0.0
    for i in psi:
        normalizer += abs(i) ** 2
    normalizer = np.sqrt(normalizer)
    return list(np.array(psi) / normalizer)

def multiply(state1, state2):   # given two component states, where qubits in state1 is to the left of those in state2
    psi = []
    for i in state1:
        for j in state2:
            psi.append(i * j)
    return psi

def generate_random_state(n_qubit):     # generate a random normalized state of n_qubit qubits
    state = []
    for _ in range(2 ** n_qubit):
        state.append(complex(random.randint(-10, 10), random.randint(-10, 10)))
    return normalize(state)


# Make sure the input state is legal
def prepare_state(n, psi):
    # Check the length of the input state vector
    if len(psi) != 2 ** n:
        print("Error 0: Input state vector of illegal length. Please try again.")
        quit()
    
    # Check the normalization of the input state vector
    normalizer = 0
    for i in psi:
        normalizer += abs(i) ** 2
    if round(normalizer, 6) != 1.0:
        print("Error 1: Input state vector not properly normalized. Please try again.")
        quit()
    # print("Input state legal. All operations ready.\n")


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


# generate the MxN A matrix for a particular combination (k, n-k) (only these k numbers are needed), for state psi
# basis states of amplitude 0 are ignored
def generate_A(comb, psi, n):      
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


def round_matrix(M, n):       # function of rounding a complex matrix M up to n decimal places
    N = M.shape
    for i in range(N[0]):
        for j in range(N[1]):
            M[i, j] = complex(round(M[i, j].real, n), round(M[i, j].imag, n))
    return M


def compare_row(a, b, n, tol=0.3):        # compare elements of two rows of the same dimensions, up to n decimal places, with a tolerance fraction
    same = True
    a = round_matrix(a, n)
    b = round_matrix(b, n)
    for i in range(a.size):
        if abs(a[0, i] - b[0, i]) > min(tol * abs(b[0, i]), tol * abs(a[0, i])):
            same = False
    return same

# Test the FULL separability of psi for n qubits
# and if possible, do the separation until none of its components is further separable
def separate(n, psi, tol=0.7):
    states = []  
    for k in range(1, n // 2 + 1):      # same for both even and odd n
        combs = generate_combs(n, k)
        for comb in combs:      # for a specific H matrix
            separable = True       # whether or not this specific H configuration is separable
            zero_rows = []          # list of zero rows, a currently useless variable
            divider_col = -1        # the column number of the divider entry in each row
            legal_row_counter = 0   # counts the number of legal (zero or identical) rows
            result = generate_A(comb, psi, n)
            A = result[2]
            H = A.getH() * A
            N = H.shape[0]
            sample_row = np.matrix(np.zeros(N))        # THE row to be compared with
            # Do the zero-entry check along with the identical-row check
            H_round = round_matrix(H, 8)    # round every element
            print("The dimension of H is ", N, file=print_f)
            for i in range(N):      # every new row     
                zeros_counter = 0
                max_value = 0
                max_index = -1
                for j in range(N):  # every element of this row
                    if H_round[i, j] == 0.0:
                        zeros_counter += 1
                    elif abs(H[i, j]) > max_value:
                        max_value = abs(H[i, j])
                        max_index = j
                if zeros_counter == N:      # a row of zeros spotted!
                    zero_rows.append(i)
                    legal_row_counter += 1
                    print("O", file=print_f)
                elif 0 < zeros_counter < N:     # anomaly detected!
                    separable = False
                    print("?", file=print_f)
                    break
                else:       # nonzero row detected! Ready for identical-row check...
                    if divider_col == -1:       # if divider column not assigned yet
                        legal_row_counter += 1
                        print("U", file=print_f)
                        divider_col = max_index
                        sample_row = H[i] / max_value
                    else:
                        identical_row = compare_row(sample_row, H[i] / H[i, divider_col], 5)        # compare two rows
                        if identical_row:
                            legal_row_counter += 1
                            print("I", file=print_f)
            # print(legal_row_counter / N)
            if legal_row_counter < tol * N:        # if not enough number of rows are legal
                separable = False

            if separable:
                # Here, this combination is confirmed to be fully separable, only print out the combination for now
                # print("Hooray! The input state is fully separable! The combination of k qubits is:")
                # print(comb)
                return True
    
    # Here, every possible configuration has been tried to no success
    # print("Sadly the input state is not separable anyhow.")
    return False


# Randomly generate a separable psi consisting of n_states multiplied together, for n qubits
def generate_separable_psi(n, n_states):
    n_qubits = []
    psi = []        # the total n-qubit state psi to be returned
    states = []     # the list of all component states, order is from LSB to MSB
    index_tool = []     # a binary operation tool used to generate psi
    temp_sum = 0
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
        index_tool.append((2 ** n_qubit - 1) * (2 ** temp_sum))     # each element is a binary number 000...000111...111100... 
                                                                    # where the positions of these 1s correspond to qubits of this state

        for j in range(2 ** n_qubit):
            new_state.append(complex(random.uniform(-10, 10), random.uniform(-10, 10)))
        # Now we normalize the new state
        normalizer = 0.0
        for k in new_state:
            normalizer += abs(k) ** 2
        normalizer = np.sqrt(normalizer)
        states.append(list(np.array(new_state) / normalizer))
        temp_sum += n_qubit

    # print(n_qubits)
    # Now we generate psi from these states   
    for index in range(2 ** n):
        product = 1.0
        processed_digits = 0
        for i in range(n_states):
            sub_index = (index & index_tool[i]) // (2 ** processed_digits)      # divide to remove 0 bits at the end
            product *= states[i][sub_index]
            processed_digits += int(np.log2(len(states[i])))
        psi.append(product)
    
    return [psi, states]

''' The main algorithm
    Outline: use psi to generate the A matrix, hence the H matrix for each
    bi-partition (k, n-k) of n qubits.
       - For odd n: test k in increasing order from 1 to (n-1)/2. In each
           test, these k qubits are a random COMBINATION, creating a total
           number of nCk partitions to test for each k.
       - For even n: test k in increasing order from 1 to n/2. Everything is
           the same, except for k=n/2, only half the partitions are needed
           since the other half partitions are equivalent to the first half.
       - To create the A matrix from a particular combination,
'''



def main():
    start = time.time()
    n = 10   # The total number of qubits
    n_states = 3    # The total number of component states
    n_trials = 50  # The total number of trials
    acc = 0
    separable_states = []   # List of state configurations that have been CORRECTLY IDENTIFIED to be separable
    # psi = list(np.array([1] * 1024) / 32)
    for _ in range(n_trials):
        [psi, states] = generate_separable_psi(n, n_states)  # random separable input state
        prepare_state(n, psi)
        if separate(n, psi):
            state_config = []
            for state in states:                                # Appends the state configuration that has been CORRECTLY IDENTIFIED
                state_config.append(int(np.log2(len(state))))
            separable_states.append(state_config)
            acc += 1
    acc /= n_trials
    end = time.time()
    interval = round(end - start, 2)
    print("The accuracy for {} trials on {} qubits with {} separable states is {}.".format(n_trials, n, n_states, acc))
    print("The whole process took {} seconds".format(interval))
    print("Separable states:\n", separable_states)
    # print("The original states are ", states)
    print_f.close()



def test2():
    x = ['A', 'B', 'C', 'D', 'E']
    a1 = np.arange(len(x)) * 100
    a2 = np.arange(len(x)) / 100
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('a1', color='red')
    X_axis = np.arange(len(x))
    plt.xticks(X_axis, x)

    ax2 = ax.twinx()
    ax2.set_ylabel('a2', color='blue')
    ax.bar(X_axis - 0.2, a1, width=0.4, color='red')
    ax2.bar(X_axis + 0.2, a2, width=0.4, color='blue')
    plt.show()

def test(n_trials=50):
    situations = [[1, 1], [1, 2], [1, 3], [2, 2], [1, 4], [2, 3], [1, 5], [2, 4], [3, 3], [1, 6], [2, 5], [3, 4], 
                    [1, 7], [2, 6], [3, 5], [4, 4], [1, 8], [2, 7], [3, 6], [4, 5], [1, 9], [2, 8], [3, 7], [4, 6], [5, 5]]     # situations to be tested
    accuracies = []
    time_taken = []
    process = 0.0
    for situation in situations:        
        n = situation[0] + situation[1]
        acc = 0
        start = time.time()
        for _ in range(n_trials):
            state1 = generate_random_state(situation[0])
            state2 = generate_random_state(situation[1])
            psi = multiply(state1, state2)           
            if separate(n, psi):
                acc += 1
        end = time.time()
        time_taken.append(int(end - start))
        acc /= n_trials
        accuracies.append(acc)
        process += 1 / len(situations)
        print("{:.0%} completed".format(process))
    
    # Now plot the results   
    X_axis = np.arange(len(situations))  
    fig, ax_acc = plt.subplots()
    ax_acc.set_xlabel('Partitions of Qubits')
    ax_acc.set_ylabel('Accuracy', color='red')
    plt.title('Accuracy and efficiency of the algorithm vs partition of qubits, with {} trials'.format(n_trials))
    plt.xticks(X_axis, list(map(str, situations)))
    
    ax_eff = ax_acc.twinx()
    ax_eff.set_ylabel('Time taken (s)', color='blue')
    ax_acc.bar(X_axis - 0.2, accuracies, width=0.4, color='red')
    ax_eff.bar(X_axis + 0.2, time_taken, width=0.4, color='blue')

    plt.show()


def test3():

    return



if __name__ == '__main__':
    test()
    # main()
    print_f.close()