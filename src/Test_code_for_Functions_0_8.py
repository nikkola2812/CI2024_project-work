#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Computational Intelligence
#Labs and Project Report
#Nikola Stankovic
#Politecnico di Torino
#s334009@studenti.polito.it


import numpy as np
import math

# List of problem files
files = ['problem_0.npz','problem_1.npz','problem_2.npz','problem_3.npz','problem_4.npz','problem_5.npz','problem_6.npz','problem_7.npz','problem_8.npz']

def function_0(x):
    y = x[0] + 0.2 * np.sin(x[1])
    return y

def function_1(x):
    y = np.sin(x[0])
    return y

def function_2(x):
    y = 7.643021e+06 * np.sin((2*x[0] + x[1] + x[2])/4)
    return y

def function_3(x):
    y = 4 - 3.5 * x[2] + 2 * x[0]**2 - x[1]**3
    return y

def function_4(x):
    y = 3.279416504354730 - 1/11 * x[0] + 7 * np.cos(x[1])
    return y

def function_5(x):
    y=-0.1*x[0]**x[1]+1.624224290224829
    y= y / 1e+10 
    return y

def function_6(x):
    y = 0.694520400387519*(x[1] - x[0]) + x[1]
    return y

def function_7(x):
    xm=x[1]-x[0]
    xp=x[1]+x[0]
    y=np.log(np.abs(xm))
    x=xp
    y=((1.3117133 + (2.0538354 * ((((np.sin((y / 1.1716985) - (0.1342675 * (1.0061529 * (1.0351123 * 1.0061529)))) / 1.0022914) + 0.44840398) - y) * (((((((((1.6232266 * y) / 0.69930816) + x) - 0.78008026) * np.sin(np.exp(0.26678434 * (y * -0.90160197)) - (0.3687389 / np.cosh(x)))) - (((y + x) / 0.81578636) + (-0.5447407 * (0.11499959 * y)))) * -0.0045083454) + np.cosh(x)) + np.cosh(x))))) - (np.cosh(-0.12136203) - ((-0.00042023603 * 0.94144845) / np.exp(y)))) + (y / (((0.26970777 + ((-0.25914347 + (((1.465689 - y) / np.cosh(x)) + (0.612497 * 0.79872614))) / -0.6667237)) / np.sin((y / 1.1129313) + y)) * ((-1.1020997 + -0.970441) - 0.0116060935)))
    return y    

def function_8(x):
    y = x[0] - 2*x[1]**2 + 3*x[3]**3 - 4*x[4]**4 + 5*x[5]**5
    return y


#=======================================================================
n = 0
for file in files:  
    print("======================================================")
    print(f"Processing {file}:")
    data = np.load(file)
    x = data['x']  # Shape (i, N)  Assumes x is a 2D array where each row is a data point.
    y = data['y']  # Shape (N,)
    
    if n==0: y_fun = function_0(x)
    if n==1: y_fun = function_1(x)
    if n==2: y_fun = function_2(x)
    if n==3: y_fun = function_3(x)
    if n==4: y_fun = function_4(x)
    if n==5: y_fun = function_5(x)
    if n==6: y_fun = function_6(x)
    if n==7: y_fun = function_7(x)
    if n==8: y_fun = function_8(x)   
        
    print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")
    print("Maximum absolute error is: ", np.max(np.abs(y-y_fun)), ", maximum abs(y): ", np.max(np.abs(y)))
    print("Maximum relative error is: ", np.max(np.abs(y-y_fun)/np.max(np.abs(y))))
    print("Mean absolute squared error", np.mean(np.abs(y_fun-y)**2))
    if np.mean(np.abs(y_fun-y)**2)==0:  print("SNR = ∞ [dB]")
    else: print("SNR = ", 10*math.log10(np.mean(np.abs(y)**2)/np.mean(np.abs(y_fun-y)**2)), "[dB]")
    print("======================================================")
    n+=1
    


# In[ ]:





# In[ ]:





# In[ ]:




