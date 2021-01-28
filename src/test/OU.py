from src.model.processes import OrnsteinUhlenbeck
from src.model.steps import ClassicalStep, PolynomialStep, LogarithmicStep
from src.model.euler import Euler
from src.test.testfunctions import test_collection
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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

############# TCL #########################
def tcl(
        nth_function=2,
        mean=0.,
        variance=0.21855,
        theta=0.5,
        mu=0,
        sigma=1,
        X0=0,
        d=1,
        alpha=2/3,
        beta=1/3,
        n=int(1e4),
        M=216
        ):
    from scipy import stats
    model = OrnsteinUhlenbeck(theta, mu, sigma, X0, d)
    coeff = PolynomialStep(alpha, beta)
    simulation = Euler(model, coeff)
    values = []
    for i in tqdm(range(M)):
        simulation.reset()
        simulation.run(n)
        values.append(simulation.nu_f(test_collection[nth_function], MODE_DISPLAY=False))
    values = np.array(values)
    std = np.sqrt(variance)
    x = np.linspace(mean-3*std, mean+3*std, 1000)
    y = stats.norm.pdf(x, loc=mean, scale=std)
    plt.hist(np.sqrt(simulation.coefficient.gamma_sum) * values, bins=int(2*M**(1/3)), density=True)
    plt.plot(x, y)
    plt.show()
