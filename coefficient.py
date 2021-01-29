import numpy as np

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

class ClassicalCoefficient(Coefficient):
    '''
    The classical coefficient with 1 / n for gamma and eta on step n.
    '''
    def __init__(self):
        gamma = lambda n : 1 / (n + 1)
        eta = lambda n : 1 / (n + 1)
        super().__init__(gamma, eta)

    def __str__(self):
        return "Classical 1/n"

class PolynomialCoefficient(Coefficient):
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
        return ("Polynomial with alpha=%.2f, beta=%.2f"%(self.alpha, self.beta))


class LogarithmicCoefficient(Coefficient):
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
        return ("Logarithmic with alpha=%.2f, beta=%.2f"%(self.alpha, self.beta))
    
class Order2PolynomialCoefficient(Coefficient):
    '''
    The Polynomial coefficient in an order 2 scheme.
    '''
    def __init__(self, alpha=1):
        gamma = lambda n : 1 / (n + 1) ** alpha
        eta = lambda n : ( gamma(n) + gamma(n+1) ) / 2
        super().__init__(gamma, eta)
        self.alpha = alpha
        
    def __str__(self):
        return ("Order 2, Polynomial with alpha=%.2f"%self.alpha)
    
class Order3PolynomialCoefficient(Coefficient):
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