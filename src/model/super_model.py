from src.test.testfunctions import test_collection
import numpy as np
import matplotlib.pyplot as plt

class Super_Model():
    def __init__(self,model,coefficient,random_seed=0):
        '''
        Init the simulation parameters. All of them are instances of classes.
        '''
        np.random.seed(random_seed)
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
        self.coefficient.reset()

    def one_step(self, step):
        raise NotImplementedError("Please Implement the one_step method")

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
            self.one_step(step)

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
