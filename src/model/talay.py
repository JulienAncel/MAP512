from src.test.testfunctions import test_collection
import numpy as np
import matplotlib.pyplot as plt
'''
Implementation of the method in G. Pages and C. Rey, 2018. Particular case of an OU process.
'''

class Talay_OU():
    def __init__(self,model,coefficient, MODE_DISPLAY=True):
        '''
        Init the simulation parameters. All of them are instances of classes.
        '''
        self.model = model
        self.coefficient = coefficient
        self.X = model.X0
        self.random = np.random.normal
        if MODE_DISPLAY:
            print("\nTesting with the Euler scheme on "+str(model)+" using "+str(coefficient)+"\n")
    
    def reset(self):
        '''
        Set up the variables in the simulation.
        step     : The index of X_n
        H        : The sum of eta_n
        invariant_law : 1/H_n sum(eta_k * sigma_(X_k-1))
        '''
        self.invariant_law = None
        self.weights = None
        self.coefficient.reset()
    
    def run(self,n):
        '''
        Run the simulation with the given parameters.
        n        : The number of steps to simulate.
        '''
        self.n = n
        self.invariant_law = np.zeros((n, self.model.d))
        self.weights = np.zeros(n)
        self.invariant_law[0] = np.copy(self.X)
        self.weights[0] = self.coefficient.eta(0)
        theta = self.model.theta
        mu = self.model.mu
        
        for step in range(1, n):
            # Some terms used in the Euler scheme formula.
            gamma_n = self.coefficient.gamma(step)
            b_n = self.model.b(self.X)
            sigma_n = self.model.sigma(self.X)
            bruit_n = self.random(size=self.model.d)
            Ab_n = self.model.Ab(self.X)
            
            if self.model.d > 1:
                sigma_tilde_n = np.zeros((self.model.d,self.model.d))
                for j in range(self.model.d):
                    for i in range(self.model.d):
                        somme_ligne = np.sum(sigma_n[:,i])
                        sigma_tilde_n[i,j] = somme_ligne * b_n[j] - theta * sigma_n[j,i] *(mu[j] + 1 - self.X[j] )  
            if self.model.d == 1:
                sigma_tilde_n = -theta * sigma_n
            
            # The derive and the diffusion terms.
            derive = gamma_n * b_n
            derive2 = gamma_n**2 * Ab_n
            diffusion = np.sqrt(gamma_n) * sigma_n * bruit_n
            diffusion2 = gamma_n**(3/2) * sigma_tilde_n * bruit_n
            # Update X
            self.X += derive + diffusion + derive2 + diffusion2

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
    
    def test_functions(
            self, 
            nb_functions_to_test = list(range(len(test_collection))),
            MODE_DISPLAY = False):
        nu_f_values = []
        for i in nb_functions_to_test:
            nu_f_values.append(self.nu_f(test_collection[i], MODE_DISPLAY))


        if MODE_DISPLAY:
            plt.legend(loc="best")
        return np.array(nu_f_values)


    