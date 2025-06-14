#!/usr/bin/env python
# coding: utf-8


#Computational Intelligence
#Labs and Project Report
#Nikola Stankovic
#Politecnico di Torino
#s334009@studenti.polito.it

import numpy as np
import math

def f0(x: np.ndarray)->np.ndarray:
    y = x[0] + 0.2 * np.sin(x[1])
    return y

def f1(x: np.ndarray)->np.ndarray:
    y = np.sin(x[0])
    return y

def f2(x: np.ndarray)->np.ndarray:
    y = 7.643021e+06 * np.sin((2*x[0] + x[1] + x[2])/4)
    return y

def f3(x: np.ndarray)->np.ndarray:
    y = 4 - 3.5 * x[2] + 2 * x[0]**2 - x[1]**3
    return y

def f4(x: np.ndarray)->np.ndarray:
    y = 3.279416504354730 - 1/11 * x[0] + 7 * np.cos(x[1])
    return y

def f5(x: np.ndarray)->np.ndarray:
    y=-0.1*x[0]**x[1]+1.624224290224829
    y= y / 1e+10 
    return y

def f6(x: np.ndarray)->np.ndarray:
    y = 0.694520400387519*(x[1] - x[0]) + x[1]
    return y

def f7(x: np.ndarray)->np.ndarray:
    xm=np.abs(x[1]-x[0])
    xp=np.abs(x[1]+x[0])
    y =np.cosh(xp)/(np.pi/100+np.tanh(xm)) + 0.329468875416961
    return y

def f8(x: np.ndarray)->np.ndarray:
    y = x[0] - 2*x[1]**2 + 3*x[3]**3 - 4*x[4]**4 + 5*x[5]**5
    return y





