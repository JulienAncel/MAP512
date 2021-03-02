from src.model.processes import MultipleInvariantMeasure
from src.model.steps import PolynomialStep, PolynomialShiftedStep
from src.model.euler import Euler
from src.test.testfunctions import test_collection
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

l = len(test_collection)
nb_functions=list(range(l))

def histogram_test(X0=1, c=0.5, n=int(1e5), shift=0):
    model = MultipleInvariantMeasure(X0, c)
    #coeff = PolynomialStep(1/3, 1/3)
    coeff = PolynomialShiftedStep(1/3, 1/3, shift)
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
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Different invariant measures test. X0={}, c={}\nnu(x)={:.2f}, nu(x^2)={:.2f}\nn={}".format(X0, c, nu_x[0], nu_x2[0],n))
    plt.show()

def limite_law(X0=1, c=0.5, n=int(1e5), M=int(1e3), shift=0):
    model = MultipleInvariantMeasure(X0, c)
    coeff = PolynomialShiftedStep(1/3, 1/3, shift)
    simulation = Euler(model, coeff)
    means = []
    for i in tqdm(range(M)):
        simulation.run(n)
        means.append(simulation.test_functions([0])[0][0])
    plt.hist(x=means,
            bins=20,
            density=True,
            )
    plt.xlabel("x_mean")
    plt.ylabel("density")
    plt.title("Histogram of the mean(x) for n = {}, M={}\n X0={}, c={}, shift={}".format(n, M, X0, c, shift))
    plt.show()

def potential():
    x = np.linspace(-5, 5, num=1000)
    def V(x):
        if abs(x) >= 3:
            return (x - 3 * np.sign(x)) ** 2
        else:
            return (x**2-9)**2 / 72
    y = []
    for i in x:
        y.append(V(i))
    y = np.array(y)
    plt.plot(x, y, label="V(x)")
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("The potential of b in the model")
    plt.show()

