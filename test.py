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

model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)
coeff = ClassicalCoefficient()
coeff = PolynomialCoefficient(1/3, 1/3)

simulation = Simulation(model, coeff)
simulation.run(n)
simulation.test_functions()