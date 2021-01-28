# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:08:03 2021

@author: Ancel
"""
import numpy as np

########## Calcul de I pour phi = 1/(1+x^2) ##########

n = int(1e7)
res=0
for i in range(n):
    X = np.random.normal(0,1)
    res += 4 * X**2 / (1 + X**2)**4

print("Intégrale I : {}".format(res/n))

########## Calcul de m pour phi = 1/(1+x^2) ##########

m = 0

print("Intégrale m : {}".format(m))

########## Calcul de hat(m) pour phi = 1/(1+x^2) ##########

def phi4(x):
    terme1 = -0.25 * x * 24 * (x-x**3) / (1+x**2)**4
    terme2 = 3/24 * 24 * (1-8*x**2 + 3*x**4) / (1+x**2)**5
    return terme1 + terme2

hat_m = 0
for i in range(n):
    X = np.random.normal(0,1)
    hat_m += phi4(X) + 0.5 * 0.25 * X**2 * ( 6*X**2 - 2) / ( 1 + X**2)**3

print("Intégrale hat_m : {}".format(-hat_m/n))
