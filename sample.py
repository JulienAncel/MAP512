import numpy as np

'''
Implementation of the method in D. Lamberton and G. Pages, 2003.
'''

    
class Model:
    def __init__(self,b,sigma,X0,d):
        '''
        Parameters
        ----------
        b : function R^d->R^d
            Drift term.
        sigma : function R^d->R^d
            Volatility term.
        X0 : vector R^d
            Initial Value - deterministic.
        d : int
            dimension.
        '''
        self.b = b
        self.sigma = sigma
        self.X0 = X0
        self.d = d

model = Model(function1,function2,init,d)
        
class Coefficient:
    def __init__(self,gamma,eta):
        '''
        Parameters
        ----------
        gamma : sequence N->R+
            Time discretization. We should have 
                                 lim(gamma_n) = 0 and sum(gamma_n) = +infty
        eta : sequence N->R+
            Weights.

        '''
        self.gamma = gamma
        self.eta = eta
        
coeff = Coefficient(gamma,eta)

class Simulation():
    def __init__(self,model,coefficient):
        '''
        Init the simulation parameters. All of them are instances of classes.
        '''
        self.model = model
        self.coefficient = coeff
    
    def reset(self):
        '''
        Set up the variables in the simulation.
        step     : The index of X_n
        H        : The sum of eta_n
        invariant_law : 1/H_n sum(eta_k * sigma_(X_k-1))
        '''
        #self.step = 1
        self.invariant_law = None
        self.probability = None
    
    def run(self,n):
        '''
        Run the simulation with the given parameters.
        n        : The number of steps to simulate.
        '''
        self.invariant_law = np.zeros((self.model.d, n))
        self.weights = np.zeros(n)
        for step in range(n):
            self.weights[step] = self.coefficient.eta(step)
            derive = self.coefficient.gamma(step) * self.model.b(self.invariant_law[step-1])
            diffusion = np.sqrt(self.model.sigma(self.invariant_law[step-1])) * np.random.normal(size=self.model.d)
            self.invariant_law[step] = self.invariant_law[step-1] +  derive + diffusion