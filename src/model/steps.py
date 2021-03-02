import numpy as np

class Step:
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
        self.gamma_function = gamma
        self.eta_function = eta
        self.reset()
    
    def reset(self):
        self.gamma_sum = 0.
        self.gamma_2_sum = 0.
        self.eta_sum = 0.

    def gamma(self, n):
        step = self.gamma_function(n)
        self.gamma_sum += step
        self.gamma_2_sum += step ** 2
        return step
    
    def eta(self, n):
        step = self.eta_function(n)
        self.eta_sum += step
        return step

class ClassicalStep(Step):
    '''
    The classical coefficient with 1 / n for gamma and eta on step n.
    '''
    def __init__(self):
        gamma = lambda n : 1 / (n + 1)
        eta = lambda n : 1 / (n + 1)
        super().__init__(gamma, eta)

    def __str__(self):
        return "classical steps with 1/n"

class PolynomialStep(Step):
    '''
    The Polynomial coefficient with 1/(n^alpha) for gamma and 1/(n^beta) for eta on step n.
    '''
    def __init__(self, alpha=1, beta=1):
        gamma = lambda n : 1 / (n + 1) ** alpha
        eta = lambda n : 1 / (n + 1) ** beta
        super().__init__(gamma, eta)
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return ("polynomial steps with alpha=%.2f, beta=%.2f"%(self.alpha, self.beta))

class PolynomialShiftedStep(Step):
    '''
    The Polynomial coefficient with 1/(n^alpha) for gamma and 1/(n^beta) for eta on step n.
    '''
    def __init__(self, alpha=1, beta=1, shift=0):
        gamma = lambda n : 1 / (n + shift + 1) ** alpha
        eta = lambda n : 1 / (n + shift + 1) ** beta
        super().__init__(gamma, eta)
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return ("polynomial steps with alpha=%.2f, beta=%.2f"%(self.alpha, self.beta))

class Order2PolynomialStep(Step):
    '''
    The Polynomial coefficient with 1/(n^alpha) for gamma and (gamma(n)+gamma(n-1))/2 for eta on step n.
    '''
    def __init__(self, alpha=1):
        gamma = lambda n : 1 / (n + 1) ** alpha
        eta = lambda n : ( gamma(n) + gamma(n+1) ) / 2
        super().__init__(gamma, eta)
        self.alpha = alpha

    def __str__(self):
        return ("polynomial steps with alpha=%.2f"%(self.alpha))

class Order3PolynomialStep(Step):
    '''
    The Polynomial coefficient in an order 3 scheme.
    '''
    def __init__(self, alpha=1):
        gamma = lambda n : 1 / (n + 1) ** alpha * ((n+1)%2==1) + 1 / n ** alpha * ((n+1)%2==0)
        eta = lambda n : ( gamma(n) + gamma(n-2) ) / 3 * (n%2==1) + 4 * gamma(n-1) / 3 * (n%2==0)
        super().__init__(gamma, eta)
        self.alpha = alpha
        
    def __str__(self):
        return ("Order 3, Polynomial with alpha=%.2f"%self.alpha)

class LogarithmicStep(Step):
    '''
    The logarithmic coefficient with 1/(log(n)^alpha) for gamma and 1/(log(n)^beta) for eta on step n.
    '''
    def __init__(self, alpha=1, beta=1):
        gamma = lambda n : 1 / (np.log(n + 2) ** alpha)
        eta = lambda n : 1 / (np.log(n + 2) ** beta)
        super().__init__(gamma, eta)
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return ("logarithmic steps with alpha=%.2f, beta=%.2f"%(self.alpha, self.beta))
