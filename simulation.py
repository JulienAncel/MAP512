from testfunctions import test_collection
import numpy as np
import matplotlib.pyplot as plt
'''
Implementation of the method in D. Lamberton and G. Pages, 2003.
'''
class Simulation():
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
        Run the simulation with the given parameters.
        n        : The number of steps to simulate.
        '''
        self.n = n
        self.invariant_law = np.zeros((n, self.model.d))
        self.weights = np.zeros(n)
        self.invariant_law[0] = np.copy(self.X)
        self.weights[0] = self.coefficient.eta(0)
        for step in range(1, n):
            # Some terms used in the Euler scheme formula.
            gamma_n = self.coefficient.gamma(step)
            b_n = self.model.b(self.X)
            sigma_n = self.model.sigma(self.X)

            # The derive and the diffusion terms.
            derive = gamma_n * b_n
            diffusion = np.sqrt(gamma_n) * sigma_n * self.random(size=self.model.d)
            # Update X
            self.X += derive + diffusion

            # The law approached
            self.weights[step] = self.coefficient.eta(step)
            self.invariant_law[step] = np.copy(self.X)

    def nu_f(self, f=lambda x : x):
        # Test of nu(f) = expectation of the invariant distribution.
        nu_f_n = np.zeros(self.invariant_law.shape)
        nu_f_n[0] = self.invariant_law[0]
        H = self.weights[0]
        for i in range(1, self.n):
            H += self.weights[i]
            nu_f_n[i] = nu_f_n[i-1] + self.weights[i]/H * (f(self.invariant_law[i]) - nu_f_n[i-1])
        print("nu of ({}) is equal to {}.".format(f, nu_f_n[-1]))
        plt.plot(nu_f_n, label=f)

    def test_functions(self):
        for test_function in test_collection:
            self.nu_f(test_function)
        plt.legend(loc="best")
        alpha = - np.log(self.coefficient.gamma(1)) / np.log(2)
        beta = - np.log(self.coefficient.eta(1)) / np.log(2)
        plt.title("alpha = %.2f \n beta = %.2f"%(alpha,beta))
        plt.show()
