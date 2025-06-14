#!/usr/bin/env python
# coding: utf-8


#======================================================
#function_0  with PySRRegressor
#======================================================
print("======================")
print("Data Set 0 in progress")
print("======================")

import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_0.npz'
problem = np.load(filename)
X = problem['x']  # Shape (2, N)
    
# Example data setup
x = X[0]
y = X[1]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y})
data['target'] = problem['y']


for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin', 'cos', 'exp'],  # Allowed functions
#        unary_operators = [
#        "sqrt",
#        "log",
#        "exp",
#        "sin",
#        "cos",
#        "tan",
#        "cosh",
#        "tanh",
#        "square",
#        "cube"],
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.1,  # Penalty for complexity of variables
        complexity_of_constants=0.1,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y']], data['target'])
    if np.mean((model.predict(data[['x', 'y']]) - data['target'])**2)/np.mean((data['target'])**2) < 1e-10:
        break
    
# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================



#======================================================
#function_0 with gplearn SymbolicRegressor
#======================================================
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split

filename = 'problem_0.npz'
problem = np.load(filename)
X = problem['x']  # Shape (2, N)
    
# Example data setup
x = X[0]
y = X[1]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y})
data['target'] = problem['y']


X_train, X_test, y_train, y_test = train_test_split(data[['x', 'y']], data['target'], test_size=0.1, random_state=42)

def _exp(x):
    # Clip values to prevent overflow in exponential
    return np.exp(np.clip(x, -20, 20))
exp_function = make_function(function=_exp, name='exp', arity=1)

# Adjusted SymbolicRegressor parameters
model = SymbolicRegressor(
    population_size=1000,  # Increased population size
    generations=100,       # Increased generations
    function_set=['add', 'sin', 'mul', exp_function],
    p_crossover=0.5,
    p_subtree_mutation=0.2,  # Increased mutation rate
    p_point_mutation=0.2,  # Increased mutation rate
    p_hoist_mutation=0.1,
    verbose=1,
    n_jobs=-1,
    random_state=43,
    parsimony_coefficient=0.0005 #even lower penalty
)

model.fit(X_train, y_train)

print("Best program found:")
print(model._program)
#======================================================


#======================================================
#function_1   with PySRRegressor
#======================================================
print("======================")
print("Data Set 1 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_1.npz'
problem = np.load(filename)
X = problem['x']  # Shape (2, N)
    
# Example data setup
x = X[0]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x})
data['target'] = problem['y']

for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin', 'cos', 'exp'],  # Allowed functions
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.1,  # Penalty for complexity of variables
        complexity_of_constants=0.1,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x']], data['target'])
    if np.mean((model.predict(data[['x']]) - data['target'])**2) < 1e-10:
        break

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_1  with gplearn SymbolicRegressor
#======================================================
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split

filename = 'problem_1.npz'
problem = np.load(filename)
X = problem['x']  # Shape (1, N)

# Example data setup
x = X[0]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x})
data['target'] = problem['y']

X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['target'], test_size=0.1, random_state=42)

def _exp(x):
	# Clip values to prevent overflow in exponential
	return np.exp(np.clip(x, -20, 20))
exp_function = make_function(function=_exp, name='exp', arity=1)

# Adjusted SymbolicRegressor parameters
model = SymbolicRegressor(
population_size=1000,  # Increased population size
generations=100,       # Increased generations
function_set=['add', 'sin', 'mul', exp_function],
p_crossover=0.5,
p_subtree_mutation=0.2,  # Increased mutation rate
p_point_mutation=0.2,  # Increased mutation rate
p_hoist_mutation=0.1,
verbose=1,
n_jobs=-1,
random_state=43,
parsimony_coefficient=0.0005 #even lower penalty
)

model.fit(X_train, y_train)

print("Best program found:")
print(model._program)
#======================================================

	
	
#======================================================
#function_2   with PySRRegressor.    !!!!!Data down-scaled
#======================================================
print("======================")
print("Data Set 2 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_2.npz'
problem = np.load(filename)
X = problem['x']  # Shape (3, N)
    
# Example data setup
x = X[0]
y = X[1]
z = X[2]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y, 'z': z})
data['target'] = problem['y']/10e+6
#==================
#!!!!!Data scaled
#==================

for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin'],  # Allowed functions
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.1,  # Penalty for complexity of variables
        complexity_of_constants=0.1,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y', 'z']], data['target'])
    if np.mean((model.predict(data[['x', 'y', 'z']]) - data['target'])**2) < 1e-10:
        break

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_3   with PySRRegressor
#======================================================
print("======================")
print("Data Set 3 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_3.npz'
problem = np.load(filename)
X = problem['x']  # Shape (3, N)
    
# Example data setup
x = X[0]
y = X[1]
z = X[2]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y, 'z': z})
data['target'] = problem['y']

for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
        binary_operators=['+', '-', '*', '/'],
        unary_operators=[ ],  # Allowed functions
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.2,  # Penalty for complexity of variables
        complexity_of_constants=0.2,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y', 'z']], data['target'])
    if np.mean((model.predict(data[['x', 'y', 'z']]) - data['target'])**2) < 1e-10:
        break


# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_4   with PySRRegressor
#======================================================
print("======================")
print("Data Set 4 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_4.npz'
problem = np.load(filename)
X = problem['x']  # Shape (2, N)
    
# Example data setup
x = X[0]
y = X[1]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y})
data['target'] = problem['y']

for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['cos'],  # Allowed functions
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.1,  # Penalty for complexity of variables
        complexity_of_constants=0.1,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y']], data['target'])
    if np.mean((model.predict(data[['x', 'y']]) - data['target'])**2)/np.mean((data['target'])**2) < 1e-10:
        break


# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_5   with PySRRegressor.  !!!!!Data up-scaled
#======================================================
print("======================")
print("Data Set 5 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Load the problem data
filename = 'problem_5.npz'
problem = np.load(filename)
X = problem['x']  # Shape (2, N)

# Example data setup
x = X[0]
y = X[1]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y})
data['target'] = (problem['y'])* 1e+10 / 2.8520706810421616

#==================
#!!!!!Data scaled
#==================

for _ in range(10):
    # Set up the symbolic regressor using PySR for the first 200 iterations
    model_initial = PySRRegressor(
        
        niterations=1000,  # First run for 200 iterations
        
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['exp', 'cos', 'log', 'sin'],  # Allowed functions
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,  # Penalty for addition
            '-': 0.1,  # Penalty for subtraction
            '*': 0.1,  # Penalty for multiplication
            '/': 0.1,  # Penalty for division
        },
        complexity_of_variables=0.5,  # Penalty for complexity of variables
        complexity_of_constants=0.5,  # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y']], data['target'])
    if np.mean((model.predict(data[['x', 'y']]) - data['target'])**2)/np.mean((data['target'])**2) < 1e-10:
        break


# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_6   with PySRRegressor
#======================================================
print("======================")
print("Data Set 6 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_6.npz'
problem = np.load(filename)
X = problem['x']  # Shape (2, N)
    
# Example data setup
x = X[0]
y = X[1]

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y})
data['target'] = problem['y']

for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
        binary_operators=['+', '-', '*', '/'],
        unary_operators=[ ],  # Allowed functions
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.1,  # Penalty for complexity of variables
        complexity_of_constants=0.1,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y']], data['target'])
    if np.mean((model.predict(data[['x', 'y']]) - data['target'])**2)/np.mean((data['target'])**2) < 1e-10:
        break

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_7 with PySRRegressor
#======================================================
print("======================")
print("Data Set 7 in progress")
print("======================")

import numpy as np
import pandas as pd
from pysr import PySRRegressor

for _ in range(10):

    # Load your data
    filename = 'problem_7.npz'
    
    problem = np.load(filename)
    X = problem['x']  # Shape (2, N)
    y = problem['y']  # Shape (N,)
    xp = X[0] + X[1]
    xm = X[0] - X[1]
    x_data = np.abs(xm)
        
    # Create a DataFrame for the inputs
    data = pd.DataFrame({'x': np.abs(xm), 'y': np.abs(xp)})
    #data = pd.DataFrame({'x': np.abs(xm)})
    data['target'] = y
    
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
        niterations=1000,  # Number of iterations to run
#        unary_operators = [
#        "sqrt",
#        "log",
#        "exp",
#        "sin",
#        "cos",
#        "tan",
#        "cosh",
#        "tanh",
#        "square",
#        "cube"],
        unary_operators = [
        "cosh",
        "tanh"],
        binary_operators=['+', '-', '*', '/'],
        model_selection='best',  # Select the best model based on generalization
        complexity_of_operators={
            '+': 0.1,   # Penalty for addition
            '-': 0.1,   # Penalty for subtraction
            '*': 0.1,   # Penalty for multiplication
            '/': 0.1,   # Penalty for division
        },
        complexity_of_variables=0.5,  # Penalty for complexity of variables
        complexity_of_constants=0.5,   # Penalty for complexity of constants
    )
    
    # Fit the model to the data
    model.fit(data[['x', 'y']], data['target'])
    if np.mean((model.predict(data[['x', 'y']]) - data['target'])**2)/np.mean((data['target'])**2) < 1e-10:
        break

    # Access the best equation and its score
    best_equation = model.get_best()  # Retrieves the equation with the best score
    print("Best model found:")
    print(best_equation.equation)  # Display the equation
    print("With score (loss):")
    print(best_equation.loss)  # Display the loss score
#======================================================


#======================================================
#function_8   with PySRRegressor
#======================================================
print("======================")
print("Data Set 8 in progress")
print("======================")
import numpy as np
import pandas as pd
from pysr import PySRRegressor

filename = 'problem_8.npz'
problem = np.load(filename)
X = problem['x']  # Shape (6, N)
    
# Example data setup
x = X[0]
y = X[1]
z = X[2]
w = X[3]
u = X[4]
v = X[5] 

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': x, 'y': y, 'z': z, 'w': w, 'u': u, 'v': v})
data['target'] = problem['y']

if data.shape[0] > 10000:
    data = data.sample(n=10000, random_state=42).reset_index(drop=True)

for _ in range(10):
    # Set up the symbolic regressor using PySR
    model = PySRRegressor(
	    binary_operators=["+", "*", "-"],  # Allow addition, subtraction, and multiplication using symbols
	    unary_operators=[ ],
	    model_selection='best',  # Select the best model based on generalization
	    complexity_of_operators={
	        '+': 0.1,   # Penalty for addition
	        '-': 0.1,   # Penalty for subtraction
	        '*': 0.1,   # Penalty for multiplication
	        '/': 0.9,   # Penalty for division
	    },
	    complexity_of_variables=0.5,  # Penalty for complexity of variables
	    complexity_of_constants=0.8,   # Penalty for complexity of constants
	    niterations=1000,  # Increase number of iterations for better search
    )

    # Fit the model to the data	
    model.fit(data[['x', 'y', 'z', 'w', 'u', 'v']], data['target'])
    if np.mean((model.predict(data[['x', 'y', 'z', 'w', 'u', 'v']]) - data['target'])**2) < 1e-10:
        break
        
# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


