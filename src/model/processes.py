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
        super().__init__(b, sigma, X0, d)
    
    def __str__(self):
        return "Ornstein Uhlenbeck process with dX_t = {}*({}-X_t)*dt + {}*dB_t".format(self.theta, self.mu, self.sigma_)
