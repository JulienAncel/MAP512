from src.model.processes import OrnsteinUhlenbeck
from src.model.steps import ClassicalStep, PolynomialStep, LogarithmicStep
from src.model.euler import Euler
from src.test.testfunctions import test_collection
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm

l = len(test_collection)
############# Curve with single alpha/beta ####################
def single_curve(
        nb_functions=list(range(l)),
        theta=0.5,
        mu=0,
        sigma=1,
        X0=0,
        d=1,
        n=int(1e4)
        ):
    model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)
    print("You can change the parameter functions to test with a list [0, ..., {}] with the number of the functions".format(l-1))
    def run(coeff):
        simulation = Euler(model, coeff)
        simulation.run(n)
        simulation.test_functions(nb_functions, MODE_DISPLAY=True)

    coeff = ClassicalStep()
    run(ClassicalStep())
    run(PolynomialStep(0.3, 0.3))
    run(LogarithmicStep(0.3, 0.3))
    plt.title("Curve with different coefficients")
    plt.legend(loc="best")
    plt.show()

############# HEATMAP with different alpha/beta ###############
def heatmap(
        nb_functions=list(range(l)),
        theta=0.5,
        mu=0,
        sigma=1,
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
    model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)
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
    for alpha in tqdm(grille_puissance_alpha):
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

############# Plot limit distribution TCL emprical vs theoretical ##########
############# for f = A.phi and phi(x) = 1/(1+x^2)                ##########
I = 0.21858665258974425 #MC estimation n = 1e7
m = 0 #exact
hat_m = 0.09064864714136386 #MC estimation n = 1e7
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
        label="density of N(0, 0.218587)",
        ))

#2nd case th9
test_instances.append(TCL_Test(
        alpha=2/3,
        beta=2/3,
        mean=0,
        var=I,
        rate=lambda n : np.sqrt(3) * n ** (1/6),
        label="density of N(0, 0.218587)",
        ))

#3rd case th9
test_instances.append(TCL_Test(
        alpha=1/2,
        beta=1/2,
        mean=2*np.sqrt(2)*m,
        var=I,
        rate=lambda n : n**(1/4),
        label="density of N(2sqrt(2)*m, 0.218587)",
        ))

#1st case th10
test_instances.append(TCL_Test(
        alpha=2/5,
        beta=2/5,
        mean=hat_m,
        var=I,
        rate=lambda n : 1/3*n**(2/5),
        label="density of N(hat_m, 0.218587)",
        ))

#2st case th10
test_instances.append(TCL_Test(
        alpha=1/3,
        beta=1/3,
        mean=np.sqrt(6)*hat_m,
        var=I,
        rate=lambda n : np.sqrt(3/2) * n **(1/3),
        label="density of N(sqrt(6)*hat_m, 0.218587)",
        ))

def tcl(
        test_instance=TCL_Test(),
        nth_function=2,
        theta=0.5,
        mu=0,
        sigma=1,
        X0=0,
        d=1,
        n=int(1e4),
        M=500
        ):
    model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)

    def distrib_result(coeff):
        values = []
        phi = test_collection[2]
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
    plt.title("Empirical vs theoretical result : alpha = {0:.2f}".format(alpha))
    plt.hist(empirical_res*rate, bins=20, density=True, label="empirical histogram")
    x = np.linspace(mean-3*std, mean+3*std, 1000)
    y = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, y, label=test_instance.label)
    plt.legend(loc="best")
    plt.show()
    return
