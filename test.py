from model import OrnsteinUhlenbeck
from coefficient import ClassicalCoefficient, PolynomialCoefficient, LogarithmicCoefficient
from simulation import Simulation
from testfunctions import test_collection
import matplotlib.pyplot as plt
#from tqdm import tqdm
import numpy as np
from scipy.stats import norm

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
'''
def run(coeff):
    simulation = Simulation(model, coeff)
    simulation.run(n)
    simulation.test_functions(MODE_DISPLAY=True)

coeff = ClassicalCoefficient()
#run(ClassicalCoefficient())
run(PolynomialCoefficient(2/3, 2/3))
#run(LogarithmicCoefficient(0.3, 0.3))
plt.title("Curve with different coefficients")
plt.legend(loc="best")
plt.show()
'''

############# Plot limit distribution TCL emprical vs theoretical ##########
############# for f = A.phi and phi(x) = 1/(1+x^2)                ##########

I = 0.21858665258974425 #MC estimation n = 1e7
m = 0 #exact
hat_m = 0.09064864714136386 #MC estimation n = 1e7

M = 500 #sampling size

def distrib_result(coeff):
    values = []
    phi = test_collection[2]
    for i in range(M):
        simulation = Simulation(model, coeff)
        simulation.run(n)
        values.append(simulation.nu_f(phi))
    
    return np.array(values)
    
#1er cas th9
coeff = PolynomialCoefficient(1, 1)
rate = np.sqrt(np.log(n))
theoretical_res = lambda x : norm.pdf(x,0,np.sqrt(I))
empirical_res = distrib_result(coeff)
#mean,std=norm.fit(empirical_res)

plt.figure()
plt.title("Emprical vs theoretical result : alpha = 1")
_, bins, _ = plt.hist(empirical_res*rate,bins=16,density=True)
plt.plot(bins, theoretical_res(bins),label="density of N(0, 0.218587)")
#plt.plot(bins, norm.pdf(bins, mean, std),label="density of N(%.2f, %.2f)"%(mean,std**2))
plt.legend()
plt.show()

#2ème cas th9
coeff = PolynomialCoefficient(2/3, 2/3)
rate = np.sqrt(3) * n**(1/6)
theoretical_res = lambda x : norm.pdf(x,0,np.sqrt(I))
empirical_res = distrib_result(coeff)
#mean,std=norm.fit(empirical_res)

plt.figure()
plt.title("Emprical vs theoretical result : alpha = 2/3")
_, bins, _ = plt.hist(empirical_res*rate,bins=16,density=True)
plt.plot(bins, theoretical_res(bins),label="density of N(0, 0.218587)")
#plt.plot(bins, norm.pdf(bins, mean, std),label="density of N(%.2f, %.2f)"%(mean,std**2))
plt.legend()
plt.show()

'''
#3ème cas th9
coeff = PolynomialCoefficient(1/2, 1/2)
rate = n**(1/4)
theoretical_res = lambda x : norm.pdf(x,2*np.sqrt(2)*0,np.sqrt(I))
empirical_res = distrib_result(coeff)

plt.figure()
plt.title("Emprical vs theoretical result : alpha = 1/2")
_, bins, _ = plt.hist(empirical_res*rate,bins=16,density=True)
plt.plot(bins, theoretical_res(bins),label="density of N(2sqrt(2)*m, 0.218587)")
plt.legend()
plt.show()



#1er cas th10
coeff = PolynomialCoefficient(2/5, 2/5)
rate = 1/3 * n**(2/5)
theoretical_res = lambda x : norm.pdf(x,hat_m,np.sqrt(I))
empirical_res = distrib_result(coeff)

plt.figure()
plt.title("Emprical vs theoretical result : alpha = 2/5")
_, bins, _ = plt.hist(empirical_res*rate,bins=16,density=True)
plt.plot(bins, theoretical_res(bins),label="density of N(hat_m, 0.218587)")
plt.legend()
plt.show()


#2ème cas th10
coeff = PolynomialCoefficient(1/3, 1/3)
rate = np.sqrt(3/2) * n**(1/3)
theoretical_res = lambda x : norm.pdf(x, np.sqrt(6) * hat_m, np.sqrt(I))
empirical_res = distrib_result(coeff)

plt.figure()
plt.title("Emprical vs theoretical result : alpha = 1/3")
_, bins, _ = plt.hist(empirical_res*rate,bins=16,density=True)
plt.plot(bins, theoretical_res(bins),label="density of N(sqrt(6)*0.0906, 0.218587)")
plt.legend()
plt.show()
'''



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
