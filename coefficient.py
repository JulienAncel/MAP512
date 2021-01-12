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

class PolynomialCoefficient(Coefficient):
    '''
    The Polynomial coefficient with 1/(n^alpha) for gamma and 1/(n^beta) for eta on step n.
    '''
    def __init__(self, alpha=1, beta=1):
        gamma = lambda n : 1 / (n + 1) ** alpha
        eta = lambda n : 1 / (n + 1) ** beta
        super().__init__(gamma, eta)
