#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#function_0 
#======================================================
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
model.fit(data[['x', 'y']], data['target'])

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score



#function_0 with gplearn
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
    generations=200,       # Increased generations
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


#function_1 
#======================================================
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

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score

#======================================================


#function_2 
#======================================================
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
model.fit(data[['x', 'y', 'z']], data['target'])

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score

#======================================================


#function_3 
#======================================================
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

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score

#======================================================


#function_4 
#======================================================
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


# Set up the symbolic regressor using PySR
model = PySRRegressor(
    niterations=100,  # Number of iterations to run
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

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score

#======================================================


#function_5 
#======================================================
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

# Set up the symbolic regressor using PySR for the first 200 iterations
model_initial = PySRRegressor(
    
    niterations=20000,  # First run for 200 iterations
    
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

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score


#======================================================


#function_6 
#======================================================
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


# Set up the symbolic regressor using PySR
model = PySRRegressor(
    niterations=100,  # Number of iterations to run
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

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score

#======================================================


#function_7 with PySRRegressor
#======================================================
import numpy as np
import pandas as pd
from pysr import PySRRegressor

# Load your data
filename = 'problem_7.npz'

problem = np.load(filename)
X = problem['x']  # Shape (2, N)
y = problem['y']  # Shape (N,)
xp = X[0] + X[1]
xm = X[0] - X[1]
x_data = np.abs(xm)

#C = np.log((y - 0.33) / (31 * np.cosh(1.0 * xp)))

# Create a DataFrame for the inputs
data = pd.DataFrame({'x': np.abs(xp), 'y': np.log(np.abs(xm))})
data['target'] = y

# Set up the symbolic regressor using PySR
model = PySRRegressor(
    niterations=10000,  # Number of iterations to run
    binary_operators=['+', '-', '*', '/'],
    unary_operators=['sin', 'exp', 'log', 'cosh'],  # Allowed functions
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
model.fit(data[['x', 'y']], data['target'])

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score
#======================================================


#function_8 
#======================================================
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

constraints = {
    'c0': (0, 5),  # Example of a constant c0 ranging from 0 to 5
}


# Set up the symbolic regressor using PySR
model = PySRRegressor(
    binary_operators=["+", "*", "-"],  # Allow addition, subtraction, and multiplication using symbols
    unary_operators=[ ],
    model_selection='best',  # Select the best model based on generalization
    complexity_of_operators={
        '+': 0.1,   # Penalty for addition
        '-': 0.1,   # Penalty for subtraction
        '*': 0.1,   # Penalty for multiplication
        '/': 0.1,   # Penalty for division
    },
    complexity_of_variables=0.001,  # Penalty for complexity of variables
    complexity_of_constants=0.001,   # Penalty for complexity of constants
    niterations=5000,  # Increase number of iterations for better search
    constraints=constraints
)

# Fit the model to the data
model.fit(data[['x', 'y', 'z', 'w', 'u', 'v']], data['target'])

# Access the best equation and its score
best_equation = model.get_best()  # Retrieves the equation with the best score
print("Best model found:")
print(best_equation.equation)  # Display the equation
print("With score (loss):")
print(best_equation.loss)  # Display the loss score


#Checking the result by polynomial Regression
#======================================================
problem = np.load('problem_8.npz')
x = problem['x']  # Shape (6, N)
y = problem['y']  # Shape (N,)
print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")

X = x.T
X_poly = X  # Create polynomial features
poly = PolynomialFeatures(degree=5)  # Specify the polynomial degree (adjust degree as needed)
X_poly = poly.fit_transform(X)  # Create polynomial features
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly) 
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Maximum prediction error:", np.max(np.abs(y_pred - y)))
coefficients = model.coef_
feature_names = poly.get_feature_names_out(input_features=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
print("Coefficients:")

for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef}")
# Optionally, set a threshold to remove insignificant terms
threshold = 0.1  # Example threshold
significant_features = [(name, coef) for name, coef in zip(feature_names, coefficients) if abs(coef) >= threshold]

# Print significant features
print("\nSignificant Features:")
for name, coef in significant_features:
    print(f"{name}: {coef}")

#======================================================


# In[ ]:







# In[ ]:




