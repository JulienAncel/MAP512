from model import OrnsteinUhlenbeck
from coefficient import ClassicalCoefficient, PolynomialCoefficient, LogarithmicCoefficient
from simulation import Simulation
from testfunctions import test_collection
import matplotlib.pyplot as plt
from tqdm import tqdm
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

n = int(1e4)

model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)

############# Curve with single alpha/beta ####################
def run(coeff):
    simulation = Simulation(model, coeff)
    simulation.run(n)
    simulation.test_functions(MODE_DISPLAY=True)

coeff = ClassicalCoefficient()
run(ClassicalCoefficient())
run(PolynomialCoefficient(0.3, 0.3))
run(LogarithmicCoefficient(0.3, 0.3))
plt.title("Curve with different coefficients")
plt.legend(loc="best")
plt.show()

############# HEATMAP with different alpha/beta ###############
'''
alpha_poly_min = 0.11
alpha_poly_sup = 0.5
nb_division_alpha = 10
grille_puissance_alpha = np.linspace(alpha_poly_min, alpha_poly_sup, num=nb_division_alpha)
beta_poly_min = 0.11
beta_poly_sup = 0.5
nb_division_beta = 10
grille_puissance_beta = np.linspace(beta_poly_min, beta_poly_sup, num=nb_division_beta)

values_list = []
for alpha in tqdm(grille_puissance_alpha):
    values_with_same_alpha = []
    for beta in grille_puissance_beta:
        coeff = PolynomialCoefficient(alpha, beta)
        simulation = Simulation(model, coeff)
        simulation.run(n)
        values_with_same_alpha.append(simulation.test_functions())
    values_list.append(values_with_same_alpha)

values_list = np.array(values_list)

def show_heatmap(nth, mean):
    plt.imshow(np.abs(values_list[:, :, nth] - mean))
    plt.colorbar()
    plt.xticks(ticks=[i for i in range(nb_division_beta)], labels=np.round(grille_puissance_beta, 2))
    plt.yticks(ticks=[i for i in range(nb_division_alpha)], labels=np.round(grille_puissance_alpha, 2))
    plt.title("Heatmap with different alpha-beta with the function = {}".format(test_collection[nth]))
    plt.show()

show_heatmap(2, 0.0)
'''
