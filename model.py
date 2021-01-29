import numpy as np

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

class OrnsteinUhlenbeck(Model):
    '''
    The Ornstein-Uhlenbeck model with the formula :
        dXt = theta*(mu - Xt)dt + sigma*dWt
    '''
    def __init__(self, theta, mu, sigma_, X0, d):
        b = lambda x: theta * (mu - x)
        sigma = lambda x: sigma_
        self.derivee_sigma = lambda x : lambda t : 0 #pour ordre 2
        self.Ab = lambda x : theta**2 * x
        super().__init__(b, sigma, X0, d)
