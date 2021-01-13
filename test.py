from model import OrnsteinUhlenbeck
from coefficient import ClassicalCoefficient, PolynomialCoefficient
from simulation import Simulation
import numpy as np

theta = 0.5
mu = 0
sigma = 1

# d = 2
X0 = np.array([0., 0.])
d = 2

# d = 1
X0 = 0
d = 1

n = int(1e3)

alpha_poly_sup = 1
beta_poly_sup = 1
grille_puissance = np.linspace(0.1, 1, num=10)

model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)
coeff = ClassicalCoefficient()

for alpha in grille_puissance:
    for beta in grille_puissance:
        coeff = PolynomialCoefficient(alpha, beta)

        simulation = Simulation(model, coeff)
        simulation.run(n)
        simulation.test_functions()