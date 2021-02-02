from src.model.super_model import Super_Model
import numpy as np
'''
Implementation of the method in G. Pages and C. Rey, 2018. Particular case of an OU process.
'''

class Talay_OU(Super_Model):
    def __init__(self,model,coefficient,random_seed=0,MODE_DISPLAY=True):
        '''
        Init the simulation parameters. All of them are instances of classes.
        '''
        super().__init__(model, coefficient,random_seed)
        if MODE_DISPLAY:
            print("\nTesting with the Euler scheme on "+str(model)+" using "+str(coefficient)+"\n")

    def one_step(self, step):
        # Some constants:
        d = self.model.d
        theta = self.model.theta
        mu = self.model.mu
        # Some terms used in the Euler scheme formula.
        gamma_n = self.coefficient.gamma(step)
        b_n = self.model.b(self.X)
        sigma_n = self.model.sigma(self.X)
        bruit_n = self.random(size=d)
        Ab_n = self.model.Ab(self.X)
        
        if d > 1:
            sigma_tilde_n = np.zeros((d,d))
            for j in range(d):
                for i in range(d):
                    somme_ligne = np.sum(sigma_n[:,i])
                    sigma_tilde_n[i,j] = somme_ligne * b_n[j] - theta * sigma_n[j,i] *(mu[j] + 1 - self.X[j] )  
        if d == 1:
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
    
