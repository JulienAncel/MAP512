import numpy as np

class TestFunction:
    def __init__(self, function, name):
        self.function = function
        self.name = name
    
    def __call__(self, x):
        return self.function(x)

    def __str__(self):
        return self.name

test_collection = []
#0
test_collection.append(TestFunction(
    lambda x : x,
    "x")
    )

#1
test_collection.append(TestFunction(
    lambda x : x **2,
    "x^2")
    )

#2
test_collection.append(TestFunction(
    lambda x : 1 / (1 + x**2) + 2 / (1 + x**2) ** 2 - 4 / (1 + x**2) ** 3 ,
    "A(1 / (1 + x**2) )")
    )

#3 Cas loi de cauchy (1,0)
test_collection.append(TestFunction(
    lambda x : (4*x-2*x**2-2)/(1+x**2)**3 - 4*x/((1+x**2)**4 + 4*x**2) ,
    "A(1 / (1 + x**2) )")
    )

#4 Cas loi hyperbolique secante
test_collection.append(TestFunction(
    lambda x : (6*x**2 - 2) / (1+x**2)**3 - 0.5*np.pi * np.tanh(np.pi * x / (1+x**2)**2 ),
    "A(1 / (1 + x**2) )")
    )

#4 Cas loi logistique(0,1)
test_collection.append(TestFunction(
    lambda x : (6*x**2 - 2) / (1+x**2)**3 - np.tanh(x / (1+x**2)**2 ),
    "A(1 / (1 + x**2) )")
    )
'''
#3

test_collection.append(TestFunction(
    lambda x : np.cos(x),
    "cos_tronc")
    )


#4

test_collection.append(TestFunction(
    lambda x : np.sin(x) * (x > -4 and x < 4),
    "sin_tronc")
    )


test_collection.append(TestFunction(
    lambda x : np.exp(x) * (x > -2 and x < 2),
    "exp_tronc")
    )

#5
test_collection.append(TestFunction(
    lambda x : np.exp(x) * (x > -2 and x < 2),
    "exp_tronc")
    )
'''
