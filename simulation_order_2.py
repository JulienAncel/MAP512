# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:31:25 2021

@author: Ancel
"""


from testfunctions import test_collection
import numpy as np
import matplotlib.pyplot as plt
'''
Implementation of the method of order 2 in G. Pages and C. Rey, 2017.
'''
class Simulation2():
    def __init__(self,model,coefficient):
        '''
        Init the simulation parameters. All of them are instances of classes.
        '''
        self.model = model
        self.coefficient = coefficient
        self.X = model.X0
        self.random = np.random.normal
    
    def reset(self):
        '''
        Set up the variables in the simulation.
        step     : The index of X_n
        H        : The sum of eta_n
        invariant_law : 1/H_n sum(eta_k * sigma_(X_k-1))
        '''
        self.invariant_law = None
        self.weights = None
        
    def run(self,n):
        '''
        Run the simulation with the given parameters and an order 2 
        approximation of Xt.
        n        : The number of steps to simulate.
        '''
        self.n = n
        self.invariant_law = np.zeros((n, self.model.d))
        self.weights = np.zeros(n)
        self.invariant_law[0] = np.copy(self.X)
        self.weights[0] = self.coefficient.eta(0)
        for step in range(1, n):
            # Some terms used in the approximation scheme formula.
            gamma_n = self.coefficient.gamma(step)
            b_n = self.model.b(self.X)
            sigma_n = self.model.sigma(self.X)
            derivee_sigma_n = self.model.derivee_sigma(self.X) #une fonction
            gene_derive_n = self.model.Ab(self.X)
            bruit_n = self.random(size=self.model.d)
            
            W_n = np.zeros((self.model.d,self.model.d))
            k_n = np.random.binomial(1,0.5,(self.model.d,self.model.d)) - 0.5
            for i in range(self.model.d):
                for j in range(self.model.d):
                    if i == j:
                        W_n[i,i] = bruit_n[i] ** 2 - 1
                    else:
                        W_n[i,j] = bruit_n[i] * bruit_n[j] - k_n[min(i,j),max(i,j)]
            
            sigma_tilde_n = np.zeros((self.model.d,self.model.d))
            for i in range(self.model.d):
                inter = gradient_b_n*sigma_n[i] + gradient_sigma_n[i] * b_n + ??????
                sigma_tilde_n[i,:] = np.sum(inter)
                    
            
            ################################# A MODIF #############################
            # The derive and the diffusion terms.
            derive = gamma_n * b_n
            diffusion = np.sqrt(gamma_n) * sigma_n.dot( bruit_n )
            derive2 = gamma_n * derivee_sigma_n(np.dot( sigma_n, W_n.T) )
            diffusion2 = gamma_n ** (3/2) * sigma_tilde_n.dot( bruit_n ) 
            terme_A = gamma_n ** 2 * self.Ab(self.X)
            # Update X
            self.X += derive + derive2 + diffusion + diffusion2 + terme_A
            #######################################################################

            # The law approached
            self.weights[step] = self.coefficient.eta(step)
            self.invariant_law[step] = np.copy(self.X)
            
    def nu_f(self, f=lambda x : x, MODE_DISPLAY = False):
        # Test of nu(f) = expectation of the invariant distribution.
        nu_f_n = np.zeros(self.invariant_law.shape)
        nu_f_n[0] = self.invariant_law[0]
        H = self.weights[0]
        for i in range(1, self.n):
            H += self.weights[i]
            nu_f_n[i] = nu_f_n[i-1] + self.weights[i]/H * (f(self.invariant_law[i]) - nu_f_n[i-1])
            
        if MODE_DISPLAY:
            print("nu of ({}) is equal to {}.".format(f, nu_f_n[-1]))
            #plt.plot(nu_f_n, label=str(f))
            plt.plot(nu_f_n, label=str(f)+" with "+str(self.coefficient))
        return nu_f_n[-1]

    def test_functions(self, MODE_DISPLAY = False):
        nu_f_values = []
        for test_function in test_collection:
            nu_f_values.append(self.nu_f(test_function, MODE_DISPLAY))

        if MODE_DISPLAY:
            plt.legend(loc="best")
        return np.array(nu_f_values)