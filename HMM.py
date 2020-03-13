########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
#        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
#        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2A)
        ###
        ###
        ###
        
        T1 = np.zeros((self.L,len(x)))
        T2 = np.zeros((self.L, len(x)))
        for i in range(self.L):
            T1[i,0] = self.A_start[i] * self.O[i][x[0]] # assume initial state follows unif
        T2[:,0] = 0
        for j in range(1,len(x)):
            for i in range(self.L):
                prob_mat = np.zeros((self.L))
                for k in range(self.L):
                    prob_mat[k] = T1[k,j-1] * self.A[k][i] * self.O[i][x[j]]
                T1[i,j] = np.max(prob_mat)
                T2[i,j] = np.argmax(prob_mat)
        #print(T1)
        #print(T2)
        z = np.zeros(len(x))
        X = np.zeros(len(x))
        z[len(x)-1] = np.argmax(T1[:,len(x)-1])
        X[len(x)-1] = np.argmax(T1[:,len(x)-1])
        
        for i in range(1,len(x)):
            z[len(x)-i-1] = T2[int(z[len(x)-i])][len(x)-i]
        #print(z)
        max_seq = ''
        for digit in z:
            max_seq += str(int(digit))
        #print(max_seq)
        '''
        probs = np.zeros((self.L,len(x)+1))
        for i in range(self.L):
            probs[i][0] = 1/self.L * self.O[i][x[0]]
        seqs = np.zeros((self.L,len(x)+1))
        seqs[:,0] = 0
        for j in range(1,len(x)):
            for i in range(self.L):
                prob_mat = np.zeros((self.L))
                for k in range(self.L):
                    prob_mat[k] = probs[k][j-1] * self.A[k][i] * self.O[i][x[j-1]]
                probs[i][j] = np.max(prob_mat)
                seqs[i][j] = np.argmax(prob_mat)
        print(probs)
        print(seqs)
        z = np.zeros(len(x))
        X = np.zeros(len(x))
        z[len(x)-1] = np.argmax(probs[:,len(x)-1])
        X[len(x)-1] = np.argmax(probs[:,len(x)-1])
        for i in range(1,len(x)):
            z[len(x)-i-1] = seqs[int(z[len(x)-i])][len(x)-i]
        max_seq = ''
        for digit in z:
            max_seq += str(int(digit))
        '''
        return max_seq 


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M+1)] # shape = [M+1, L]

        ### TODO: Insert Your Code Here (2Bi)
        emi_mat = np.asarray(self.O)
        # emi_mat = np.reshape(emi_mat,(self.L,self.D))
        trans_mat = np.asarray(self.A)
        trans_mat = np.reshape(trans_mat,(self.L,self.L))

        alphas[1][:] = self.A_start * emi_mat[:,x[0]]

        for i in range(2,M+1):
            for j in range(self.L):
                summation = np.dot(alphas[i-1],trans_mat[:,j])
                alphas[i][j] = emi_mat[j][x[i-1]] * summation

            if normalize:
                sum_row = sum(alphas[i])
                
                for j in range(self.L):
                    alphas[i][j] /= sum_row
        
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bii)
        ###
        ###
        ###
        for i in range(self.L):
            betas[M][i] = 1

        emi_mat = np.asarray(self.O)
        emi_mat = np.reshape(emi_mat,(self.L,self.D))

        for i in range(M-1,0,-1):
            for j in range(self.L):
                betas[i][j] = np.dot(self.A[j][:],(np.array(betas[i+1]) * emi_mat[:,x[i]]))

            if normalize:
                sum_row = sum(betas[i])
                for j in range(self.L):
                    betas[i][j] /= sum_row

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        ###
        ###
        ###
        for a in range(self.L):
            for b in range(self.L):
                #numerator
                numerator = 0
                denominator = 0
                for i2 in range(len(Y)): #each seq
                    for i3 in range(len(Y[i2])-1): #within a seq
                        if Y[i2][i3+1] == b and Y[i2][i3] == a:
                            numerator += 1
                #denominator
                for i2 in range(len(Y)):
                    for i3 in range(len(Y[i2])-1):
                        if Y[i2][i3] == a:
                            denominator += 1
                self.A[a][b] = numerator/denominator
        
        for i in range(self.L):
            for j in range(self.D):
                numerator = 0
                denominator = 0
                for i2 in range(len(Y)):
                    for i3 in range(len(Y[i2])):
                        if X[i2][i3] == j and Y[i2][i3] == i:
                            numerator += 1
                for i2 in range(len(Y)):
                    for i3 in range(len(Y[i2])):
                        if Y[i2][i3] == i:
                            denominator += 1
                self.O[i][j] = numerator/denominator
        # Calculate each element of O using the M-step formulas.

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        ###
        ###
        ###
    
        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2D)
        ###
        ###
        ###
        emi_mat = np.asarray(self.O)
        emi_mat = np.reshape(emi_mat,(self.L,self.D))
        trans_mat = np.asarray(self.A)
        trans_mat = np.reshape(trans_mat,(self.L,self.L))

        for i in range(N_iters):
            print(i)
            # initialise the nums and dens of self.A and self.O
            temp_A_denom = np.zeros((self.L))
            temp_A_numer = np.zeros((self.L,self.L))
            temp_O_denom = np.zeros((self.L))
            temp_O_numer = np.zeros((self.L,self.D))
            
            for j in range(len(X)):
                # calc alphas and betas and convert to arrays
                alpha = self.forward(X[j],True)
                # alpha = np.reshape(np.asarray(alpha),(len(X[j]),self.L))
                #alpha = alpha/alpha.sum(axis=1,keepdims=1)
                beta = self.backward(X[j],True)
                #beta = np.reshape(np.asarray(beta),(len(X[j]),self.L))
                #beta = beta/beta.sum(axis=1,keepdims=1)
                # gamma


                gamma = np.zeros((len(X[j])+1,self.L))
                for i2 in range(len(X[j])+1):
                    for i3 in range(self.L):
                        gamma[i2][i3] = alpha[i2][i3] * beta[i2][i3]
                for row in gamma[1:,:]:
                    norm = np.sum(row)
                    if norm > 0:
                        row /= norm

                # xi
                # TODO +1?
                xi = np.zeros((len(X[j])+1,self.L,self.L))

                for i2 in range(1,len(X[j])):
                    for i3 in range(self.L):
                        for i4 in range(self.L):
                            xi[i2][i3][i4] = alpha[i2][i3] * trans_mat[i3][i4] * beta[i2+1][i4] * emi_mat[i4][X[j][i2]]
                
                for row in xi[1:,:,:]:
                    norm = np.sum(row)
                    if norm > 0:
                        row /= norm

                for row in xi[1:]:
                    temp_A_numer += row
                for row in gamma[1:len(X[j])]:
                    temp_A_denom += row

                for i2 in range(1,len(X[j])+1):
                    temp_O_denom += gamma[i2]
                    for i3 in range(self.L):
                        temp_O_numer[i3][X[j][i2-1]] += gamma[i2][i3]


            for i in range(self.L):
                for j in range(self.L):
                    trans_mat[i][j] = temp_A_numer[i][j] / temp_A_denom[i] 

            for i in range(self.L):
                for j in range(self.D):
                    emi_mat[i][j] = temp_O_numer[i][j] / temp_O_denom[i]
            #trans_mat = temp_A_numer/temp_A_denom[:,None]
            #emi_mat = temp_O_numer/temp_A_denom[:,None]
            self.A = trans_mat.tolist()
            self.O = emi_mat.tolist()
        

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)
        ###
        ###
        '''
        states.append(random.choice(range(self.L)))
        emission.append(np.argmax(self.O[states[0]]))
        for i in range(1,M):
            states.append((np.argmax(self.A[states[i-1]])))
            emission.append(np.argmax(self.O[states[i]]))
        print(states)
        print(emission)
        '''
        tran_mat = np.asarray(self.A)
        tran_mat = np.reshape(tran_mat,(self.L,self.L))
        cumsum_A = np.zeros((self.L,self.L))
        emi_mat = np.asarray(self.O)
        emi_mat = np.reshape(emi_mat,(self.L, self.D))
        cumsum_O = np.zeros((self.L,self.D))
        for i in range(self.L):
            for j in range(self.L):
                if j == 0:
                    cumsum_A[i][j] = tran_mat[i][j]
                else:
                    cumsum_A[i][j] = cumsum_A[i][j-1] + tran_mat[i][j]
        #print(cumsum_A)
        for i in range(self.L):
            for j in range(self.D):
                if j == 0:
                    cumsum_O[i][j] = emi_mat[i][j]
                else:
                    cumsum_O[i][j] = cumsum_O[i][j-1] + emi_mat[i][j]
        #print(cumsum_O)
        states.append(random.choice(range(self.L)))
        r = random.uniform(0,1)
        k = 0
        flag2 = 0
        while k < self.D and flag2 == 0:
            if r < cumsum_O[states[0]][k]:
                emission.append(k)
                flag2 = 1
            k += 1
        for i in range(1,M): # length of sequence
            r1 = random.uniform(0,1)
            flag1 = 0
            flag2 = 0
            j = 0
            k = 0
            while j < self.L and flag1 == 0:
                if r1 < cumsum_A[states[i-1]][j]:
                    states.append(j)
                    flag1 = 1
                    r2 = random.uniform(0,1)
                    while k < self.D and flag2 == 0:
                        if r2 < cumsum_O[states[i]][k]:
                            emission.append(k)
                            flag2 = 1
                        k += 1
                j += 1
        #print(states)
        #print(emission)
        return emission, states

    def generate_emission_backwards(self, M, start_state):
        emission = []
        states = []

        i=0
        emission.append(start_state)
        
        temp = []
        for s in self.O:
            temp.append(s[start_state])
        state = temp.index(max(temp))
        states.append(state)

        for i in range(1,M):
        	state = np.random.choice(self.L,p=self.A[state])
        	states.append(state)
        	emission.append(np.random.choice(self.D,p=self.O[state]))

        return emission, states
    
    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm
    
    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
