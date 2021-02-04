# -*- coding: utf-8 -*-

import time
import numpy as np


class MLOPE:
    """
    AYDINS_NOTE: rows --> instances

    Implements ML-OPE for LDA as described in "Inference in topic models II: provably guaranteed algorithms".
    """

    def __init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer):
        """
        Arguments:
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            tau0: A (positive) learning parameter that downweights early iterations.
            kappa: Learning rate: exponential decay rate should be between
                   (0.5, 1.0] to guarantee asymptotic convergence.
            iter_infer: Number of iterations of FW algorithm 

#=====> Note that if you pass the same set of all documents in the corpus every time and    <=======
#=====> set kappa=0 this class can also be used to do batch OPE.                            <=======
        """

        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = alpha
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 1
        self.INF_MAX_ITER = iter_infer

        # Initialize beta (topics)
        self.beta = np.random.rand(self.num_topics, self.num_terms) + 1e-10
        beta_norm = self.beta.sum(axis=1)
        self.beta /= beta_norm[:, np.newaxis]


    def static_online(self, batch_size, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics in M step.
		
        Arguments:
        batch_size: Number of documents of the mini-batch.
        wordids: A list whose each element is an array (terms), corresponding to a document.
                 Each element of the array is index of a unique term, which appears in the document,
                 in the vocabulary.
        wordcts: A list whose each element is an array (frequency), corresponding to a document.
                 Each element of the array says how many time the corresponding term in wordids appears
                 in the document.
        Returns time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch.        		
        """
        # E step
        start1 = time.time()
        theta = self.e_step(batch_size, wordids, wordcts)
        end1 = time.time()
        # M step
        start2 = time.time()
        self.m_step(batch_size, wordids, wordcts, theta)
        end2 = time.time()
        return end1 - start1, end2 - start2, theta

    def e_step(self, batch_size, wordids, wordcts):
        """
        Does e step
        Returns topic mixtures theta.
        """
        #TODO: return theta below outside loop, dont initialize everytime

        # Declare theta of minibatch
            theta = np.zeros((batch_size, self.num_topics))

        # Initialize new parameters
        '''mu = np.zeros((batch_size, self.num_topics))                                # ++++++++
        #pfi = np.exp()
        user_size = 444
        e = 0.3
        f = 0.3
        lamb = 1.0
        pfi = np.ones((user_size, batch_size, self.num_topics))
        shp = np.ones((user_size, self.num_topics)) * 0.3
        rte = np.ones((user_size, self.num_topics)) * 0.3'''

        # Inference
        for d in range(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d], d)
            theta[d, :] = thetad

            #mud = self.infer_mu(theta)                                             # ++++++++
            #mu[d, :] = mud                                                         # ++++++++
        return theta

    def infer_mu(self):                                                             # ++++++++
        return 0


    def infer_doc(self, ids, cts, d):
        """
        Does inference for a document using Online MAP Estimation algorithm.
        
        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns inferred theta.
        """

        # locate cache memory
        beta = self.beta[:, ids]

        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)

        # Initialize mu - offset for theta
        '''lamb = 1
        mu = theta + np.random.normal(0, lamb, theta.shape[0])             ########++++#########
        print(theta)
        exit()
        print(mu)
        exit()
        #df = T[0] * (np.dot(beta, cts / x) + (self.alpha - 1) / theta) + T[1] * (-1 * self.lamb * (theta - mu))'''


        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)

        p = 0.5
        T_lower = [1, 0]
        T_upper = [0, 1]

        for l in range(1, self.INF_MAX_ITER):
            # ======== Lower ==========
            if np.random.rand() < p:
                T_lower[0] += 1
            else:
                T_lower[1] += 1

            G_1 = np.dot(beta, cts / x) / p  # maybe outside loop???
            G_2 = (self.alpha - 1) / theta / (1 - p)  # maybe outside loop???

            ft_lower = T_lower[0] * G_1 + T_lower[1] * G_2
            index_lower = np.argmax(ft_lower)
            alpha = 1.0 / (l + 1)
            theta_lower = np.copy(theta)
            theta_lower *= 1 - alpha
            theta_lower[index_lower] += alpha

            # ======== Upper ==========
            if np.random.rand() < p:
                T_upper[0] += 1
            else:
                T_upper[1] += 1

            ft_upper = T_upper[0] * G_1 + T_upper[1] * G_2
            index_upper = np.argmax(ft_upper)
            alpha = 1.0 / (l + 1)
            theta_upper = np.copy(theta)
            theta_upper *= 1 - alpha
            theta_upper[index_upper] += alpha

            # print(theta_upper - theta_lower)
            # ======== Decision ========
            x_l = np.dot(cts, np.log(np.dot(theta_lower, beta))) + (self.alpha - 1) * np.log(theta_lower)
            x_u = np.dot(cts, np.log(np.dot(theta_upper, beta))) + (self.alpha - 1) * np.log(theta_upper)

            compare = np.array([x_l[0], x_u[0]])
            best = np.argmax(compare)

            if best == 0:
                theta = theta_lower
                x = x + alpha * (beta[index_lower, :] - x)
            else:
                theta = theta_upper
                x = x + alpha * (beta[index_upper, :] - x)

        return theta

        '''# Loop
        T = [1, 0]
        for l in range(1, self.INF_MAX_ITER):
            # Pick fi uniformly
            T[np.random.randint(2)] += 1

            # Select a vertex with the largest value of  
            # derivative of the function F
            df = T[0] * np.dot(beta, cts / x) + T[1] * (self.alpha - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)  # why +1 ?? because, otherwise, alpha = 1/1 = 1 ==> theta *= 1 - alpha *= 0 ==> theta = 0 which is baddd
            # Update theta
            theta *= 1 - alpha    # take from all of indices
            theta[index] += alpha  # add to argmax index
            # Update x
            x = x + alpha * (beta[index, :] - x)  # EXPLORE more
        return theta'''

    def m_step(self, batch_size, wordids, wordcts, theta):
        """
        Does m step: update global variables beta.
        """
        # Compute intermediate beta which is denoted as "unit beta"
        beta = np.zeros((self.num_topics, self.num_terms), dtype=float)
        for d in range(batch_size):
            beta[:, wordids[d]] += np.outer(theta[d], wordcts[d])
        # Check zeros index
        beta_sum = beta.sum(axis=0)
        ids = np.where(beta_sum != 0)[0]
        unit_beta = beta[:, ids]
        # Normalize the intermediate beta
        unit_beta_norm = unit_beta.sum(axis=1)
        unit_beta /= unit_beta_norm[:, np.newaxis]
        # Update beta    
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self.beta *= (1 - rhot)
        self.beta[:, ids] += unit_beta * rhot
        self.updatect += 1
