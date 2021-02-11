import numpy as np


class Process:

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

class OrnsteinUhlenbeck(Process):
    '''
    The Ornstein-Uhlenbeck model with the formula :
        dXt = theta*(mu - Xt)dt + sigma*dWt
    '''
    def __init__(self, theta, mu, sigma_, X0, d):
        self.theta = theta
        self.mu = mu
        self.sigma_ = sigma_
        b = lambda x: theta * (mu - x)
        sigma = lambda x: sigma_
        self.derivee_sigma = lambda x : lambda t : np.zeros(d) #pour ordre 2
        self.Ab = lambda x : np.vdot(theta * (mu - x), -theta * np.ones(d))
        super().__init__(b, sigma, X0, d)
    
    def __str__(self):
        return "Ornstein Uhlenbeck process with dX_t = {}*({}-X_t)*dt + {}*dB_t".format(self.theta, self.mu, self.sigma_)

class MultipleInvariantMeasure(Process):
    '''
    A process with multiple invariant measure possible
    '''
    def __init__(self, X0, c_):
        '''
        X0 : the initial position
        c  : the drift value
        '''
        def b_(x):
            if abs(x) >= 3:
                return -2 * (x - 3*np.sign(x))
            else:
                return -x**3/18 + x/2
        self.b = b_
        self.c = c_
        self.sigma = lambda x : c_*x
        self.X0 = X0
        self.d = 1

    def __str__(self):
        return "A model with multiple invariant measure in -3, 0, 3 with X0={} ,c={}".format(self.X0, self.c)

