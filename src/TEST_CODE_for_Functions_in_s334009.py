#!/usr/bin/env python
# coding: utf-8


#Computational Intelligence
#Labs and Project Report
#Nikola Stankovic
#Politecnico di Torino
#s334009@studenti.polito.it


import numpy as np
import math

# Import all functions from your file with prefix sn
import s334009 as sn

# List of problem files
files = ['problem_0.npz','problem_1.npz','problem_2.npz','problem_3.npz','problem_4.npz','problem_5.npz','problem_6.npz','problem_7.npz','problem_8.npz']

n = 0
for file in files:  
    print("======================================================")
    print(f"Processing {file}:")
    data = np.load(file)
    x = data['x']
    y = data['y']
    # Call the correct function based on n
    if n==0: y_fun = sn.f0(x)
    if n==1: y_fun = sn.f1(x)
    if n==2: y_fun = sn.f2(x)
    if n==3: y_fun = sn.f3(x)
    if n==4: y_fun = sn.f4(x)
    if n==5: y_fun = sn.f5(x)
    if n==6: y_fun = sn.f6(x)
    if n==7: y_fun = sn.f7(x)
    if n==8: y_fun = sn.f8(x)
    print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")
    print("Maximum absolute error is: ", np.max(np.abs(y - y_fun)), ", maximum abs(y): ", np.max(np.abs(y)))
    print("Maximum relative error is: ", np.max(np.abs(y - y_fun)/np.max(np.abs(y))))
    print("Mean absolute squared error", np.mean(np.abs(y_fun - y)**2))
    if np.mean(np.abs(y_fun - y)**2) == 0:
        print("SNR = ∞ [dB]")
    else:
        print("SNR = ", 10*math.log10(np.mean(np.abs(y)**2)/np.mean(np.abs(y_fun - y)**2)), "[dB]")
    print("======================================================")
    n += 1



