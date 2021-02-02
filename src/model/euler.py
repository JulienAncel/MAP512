from src.model.super_model import Super_Model
import numpy as np
'''
Implementation of the method in D. Lamberton and G. Pages, 2003.
'''
class Euler(Super_Model):
    def __init__(self,model,coefficient,random_seed=0,MODE_DISPLAY=True):
        '''
        Init the simulation parameters. All of them are instances of classes.
        '''
        super().__init__(model, coefficient,random_seed)
        if MODE_DISPLAY:
            print("\nTesting with the Euler scheme on "+str(model)+" using "+str(coefficient)+"\n")

    def one_step(self, step):
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
