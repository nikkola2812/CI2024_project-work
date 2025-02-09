The task of this project is to find analytical expressions for given data sets. The data sets are of different dimensions, starting from one-dimensional problems, then most of them are two-dimensional, while the highest dimension of a problem is five. 

In general, we have tried to apply several available symbolic regression approaches to find the most suitable analytic expression. 
Available Python tools are: 
	DEAP (Distributed Evolutionary Algorithms in Python)
	PySR (Python Symbolic Regression)
	gplearn

These libraries and tools provide various methodologies for conducting symbolic regression, from dedicated solutions like PySR and gplearn to more general-purpose frameworks like DEAP.

In most cases, we have used the PySR (Python Symbolic Regression) regressor. 

The codes are given in the file: 

Data_Regression_code.py

In order to have an impression of the subset of function that will be used in symbolic regression, we have implemented the following steps:

	The data set is read,
	The dimensionality is determined,
	The graphic of data for dimensions one and two is shown.
	 In one case, the data are scaled for the analysis and scaled back in the function. 

We have also used several classical tools to check the obtained result, like parameter optimization by curve_fit form SciPy library and Polynomial Regression.

The resulting functions are given in the file: 

s334009.py

A code to read each data set and test the accuracy of the results is given in the file 

Test_code_for_Functions_0_8.py


Note: After finding the resulting coefficients, we have always tried to apply \textit{a very close simplified form} (if it is obvious), like:
	When we get: 
		(0.2 * sin(y)) - ((((-8.340031e-10 / 1.6622427) / y) * y) - x )
	
	we try to use
		0.2 * sin(y) +x 
	by neglecting the part of e-10 order lower significance. If that improved the result, we used a simplified result as the final one in the report and in the function. 
	
	When we get the result of the form
		3.2794165 - 0.09090903 * x + 7 * np.cos(y)
	we first tried to improve the result by replacing 0.09090903 with 1/11, and if that improved the result, we then tried to \textit{add additional digits} to the only remaining coefficient k=3.2794165. \textbf{Under the assumptions}: (i) the rest of the model is correct and that (ii) the data are not noisy we can easily find the remaining digits by using just one data point, for example,  x[0,0], x[1,0], and y[0] and then re-estimate k=y[0]+1/11*x[0,0]-7*x[1,0]. In this way, instead of {\color{blue}3.2794165} we get  3.279416504354730, which commonly improved the accuracy of the result to 16 digits.   
	
	Since big data sets are estimated by functions with a very small number of parameters, chances for over-fitting are extremely low; therefore in most cases, we used all data as train sets. 


More detailed explanations are given in the Report.
