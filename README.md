# MAP512 - Discretization of the ergodic theorem

Research project for MAP512 (third year at Ecole polytechnique)


# Required :
In order to run the simulations it suffices to have numpy, matplotlib and scipy.stats installed on your computer. All of them are available with pip.

# Usage :
In the folder test, after implementing a desired testfunction for the weak almost sure convergence of the empirical measure, properties of the limit measure for an Ornstein-Uhlenbeck process can be obtained by running the functions of OU.py.\\
If you want to implement your own type of process and tract similar properties, you can implement your model in src/model/processes.py following the already existing classes. An approximating scheme of order 1 is implemented in src/model/euler.py and one of order 2 in src/model/talay.py.