import os
#os.chdir("C:/Users/Ancel/Desktop/MAP512")

from src.model.processes import PotentialProcess
from src.model.steps import ClassicalStep, PolynomialStep, LogarithmicStep
from src.model.euler import Euler
from src.test.testfunctions import test_collection
import matplotlib.pyplot as plt
#from tqdm import tqdm
import numpy as np
from scipy.stats import norm

l = len(test_collection)
############# Curve with single alpha/beta ####################
def single_curve(
        nb_functions=list(range(l)),
        U=lambda x : np.log(np.pi*(1+x**2)),
        gradU=lambda x : 2*x/(1+x**2),
        X0=0,
        d=1,
        n=int(1e4)
        ):
    model = PotentialProcess(U, gradU, X0, d)
    print("You can change the parameter functions to test with a list [0, ..., {}] with the number of the functions".format(l-1))
    def run(coeff):
        simulation = Euler(model, coeff)
        simulation.run(n)
        simulation.test_functions([0,1], MODE_DISPLAY=True)

    coeff = ClassicalStep()
    run(ClassicalStep())
    run(PolynomialStep(0.3, 0.3))
    run(LogarithmicStep(0.3, 0.3))
    plt.title("Potential Cauchy - Curve with different functions")
    plt.legend(loc="best")
    plt.show()

#single_curve()
    

################ Plot limit distributions vs theoretical density #############
def histogram_test(X0=1, 
                   U=lambda x : np.log(2*np.cosh(0.5*np.pi*x)),
                   gradU=lambda x : 0.5*np.pi*np.tanh(0.5*np.pi*x),
                   d=1,
                   n=int(1e5)):
    model = PotentialProcess(U, gradU, X0, d)
    coeff = PolynomialStep(1/3, 1/3)
    simulation = Euler(model, coeff)
    simulation.run(n)
    nu_x, nu_x2 = simulation.test_functions([0, 1])

    plt.figure()
    plt.hist(x=simulation.invariant_law,
            weights=simulation.weights, 
            bins=20,
            density=True,
            range=(-5, 5),
            )
    x = np.linspace(-10,10,num=1000)
    y = np.exp(-U(x))
    plt.plot(x,y)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Invariant measure with hyperbolic secante potential. X0={},\nnu(x)={:.2f}, nu(x^2)={:.2f}\nn={} ".format(X0, nu_x[0], nu_x2[0],n))
    plt.show()

#histogram_test()

# ############# HEATMAP with different alpha/beta ###############
def heatmap(
        nb_functions=list(range(l)),
        U=lambda x : 1,
        gradU=lambda x : 0,
        X0=0,
        d=1,
        n=int(1e4),
        alpha_poly_min=0.11,
        alpha_poly_sup=0.5,
        nb_division_alpha=10,
        beta_poly_min=0.11,
        beta_poly_sup=0.5,
        nb_division_beta=10
        ):
    model = PotentialProcess(U, gradU, X0, d)
    grille_puissance_alpha = np.linspace(alpha_poly_min, alpha_poly_sup, num=nb_division_alpha)
    grille_puissance_beta = np.linspace(beta_poly_min, beta_poly_sup, num=nb_division_beta)

    def show_heatmap(nth, mean):
        plt.imshow(np.abs(values_list[:, :, nth] - mean))
        plt.colorbar()
        plt.xticks(ticks=[i for i in range(nb_division_beta)], labels=np.round(grille_puissance_beta, 2))
        plt.yticks(ticks=[i for i in range(nb_division_alpha)], labels=np.round(grille_puissance_alpha, 2))
        plt.title("Heatmap with different alpha-beta with the function = {}".format(test_collection[nth]))
        plt.show()

    values_list = []
    for alpha in grille_puissance_alpha:
        values_with_same_alpha = []
        for beta in grille_puissance_beta:
            coeff = PolynomialStep(alpha, beta)
            simulation = Euler(model, coeff)
            simulation.run(n)
            values_with_same_alpha.append(simulation.test_functions())
        values_list.append(values_with_same_alpha)
    values_list = np.array(values_list)

    for nb_function in nb_functions:
        show_heatmap(nb_function, 0.0)

#heatmap()


############# Plot limit distribution TCL emprical vs theoretical ##########
############# for f = A.phi and phi(x) = 1/(1+x^2)                ##########
############ DEPEND ON U                                         ##########
        #à jour hyperbolic secant
I = 0.37372441  #MC estimation n = 1e7 #change according to U
m = 0 #change according to U
hat_m = 0 #MC estimation n = 1e7 #change according to U
##### TCL test instances #####
class TCL_Test:
    def __init__(
            self,
            alpha=2/3,
            beta=1/3,
            mean=0.,
            var=0.21855,
            rate=lambda n : np.sqrt(3) * n ** (1/6),
            label="density of N(0, 0.218587)"):
        self.alpha = alpha
        self.beta = beta
        self.mean = mean
        self.var = var
        self.rate = rate
        self.label = label

test_instances = []
#1st case th9
test_instances.append(TCL_Test(
        alpha=1,
        beta=1,
        mean=0,
        var=I,
        rate=lambda n : np.sqrt(np.log(n)),
        label="density of N(0, I)",
        ))

#2nd case th9
test_instances.append(TCL_Test(
        alpha=2/3,
        beta=2/3,
        mean=0,
        var=I,
        rate=lambda n : np.sqrt(3) * n ** (1/6),
        label="density of N(0, I)",
        ))

#3rd case th9
test_instances.append(TCL_Test(
        alpha=1/2,
        beta=1/2,
        mean=2*np.sqrt(2)*m,
        var=I,
        rate=lambda n : n**(1/4),
        label="density of N(2sqrt(2)*m,I)",
        ))

#1st case th10
test_instances.append(TCL_Test(
        alpha=2/5,
        beta=2/5,
        mean=hat_m,
        var=I,
        rate=lambda n : 1/3*n**(2/5),
        label="density of N(hat_m, I)",
        ))

#2st case th10
test_instances.append(TCL_Test(
        alpha=1/3,
        beta=1/3,
        mean=np.sqrt(6)*hat_m,
        var=I,
        rate=lambda n : np.sqrt(3/2) * n **(1/3),
        label="density of N(sqrt(6)*hat_m, I)",
        ))

def tcl(
        test_instance=TCL_Test(),
        nth_function=6, #CHANGE ACCORDING TO U
        U=lambda x : np.abs(x) - np.log(2),
        gradU=lambda x : 1*(x>=0) - 1*(x<0),
        label = "Potential U(x) = laplace \n",
        X0=1,
        d=1,
        n=int(1e4),
        M=500
        ):
    model = PotentialProcess(U, gradU, X0, d)

    def distrib_result(coeff):
        values = []
        phi = test_collection[nth_function]
        simulation = Euler(model, coeff)
        for i in range(M):
            simulation.reset()
            simulation.run(n)
            values.append(simulation.nu_f(phi))
        return np.array(values)

    mean = test_instance.mean
    std = np.sqrt(test_instance.var)
    alpha = test_instance.alpha
    beta = test_instance.beta
    rate = test_instance.rate(n)

    coeff = PolynomialStep(alpha, beta)
    empirical_res = distrib_result(coeff)

    plt.figure()
    plt.title(label+"Empirical vs theoretical result : alpha = {0:.2f}".format(alpha))
    plt.hist(empirical_res*rate, bins=20, density=True, label="empirical histogram")
    x = np.linspace(mean-3*std, mean+3*std, 1000)
    y = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, y, label=test_instance.label)
    plt.legend(loc="best")
    plt.show()
    return

for i in range(len(test_instances)):
    tcl(test_instance=test_instances[i])

