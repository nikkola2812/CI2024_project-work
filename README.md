The task of this project is to find analytical expressions for given data sets. The data sets are of different dimensions, starting from one-dimensional problems, then most of them are two-dimensional, while the highest dimension of a problem is six. 

In general, we have tried to apply several available symbolic regression approaches to find the most suitable analytic expression. 
These libraries and tools provide various methodologies for conducting symbolic regression, from dedicated solutions like PySR and gplearn to more general-purpose frameworks like DEAP.

We used all of them and present the results with: 

	PySR (Python Symbolic Regression)
	gplearn (Symbolic Regression)

In most cases, we have used the PySR (Python Symbolic Regression) regressor. 

The codes for all 9 problems are given in one file: 

Data_SymbolicRegression_ALL_codes_in_ONE.py

In order to get an impression of the subset of function that will be used in symbolic regression, we have implemented the following pre-processing steps (not included in the code).

	The data set is read,
	The dimensionality is determined,
	The graphic of data for dimensions one and two is shown.
	In two case, the data are scaled for the analysis and scaled back in the function since the significantly differ from order of 1 (very large values or very small values). 

We have also used several classical tools to check the obtained result, like parameter optimization by curve_fit form SciPy library and Polynomial Regression (not included in the code).

The resulting functions are given in the file: 

s334009.py

And for BACKUP within: DEFINITIONS_Functions_0_to_8_Stankovic.py.

A code to read each data set and test the accuracy of the results in the previous file is given in the file 

TEST_CODE_for_Functions_in_s334009.py

It imports all functions from  file s334009.py as

from s334009 import sn

- SNR = ∞  dB, meaning the exact function is obtained, or 
- SNR > 300 dB, meaning computer precision of about 16 digits

In all considered cases the obtained precision is either exact or within computer precision of about 16 digits. 

IMPORTANT NOTE: After finding the resulting coefficients, we have always tried to apply \textit{a very close simplified form} (if it is obvious), like:
	When we get: 
		(0.2 * sin(y)) - ((((-8.340031e-10 / 1.6622427) / y) * y) - x )
	
	we try to use
		0.2 * sin(y) +x 
	by neglecting the part of 1e-10 order lower significance. If that improved the result, we used a simplified result as the final one in the report and in the function. 
	
	When we get the result, for example, of the form
		3.2794165 - 0.09090903 * x + 7 * np.cos(y)
	we first tried to improve the result by replacing 0.09090903 with 1/11, and if that improved the result, we then tried to \textit{add additional digits} to the only remaining coefficient k=3.2794165. \textbf{Under the assumptions}: 
(i) the rest of the model is correct and that 
(ii) the data are not noisy we can easily find the remaining digits by using just one data point, for example,  x[0,0], x[1,0], and y[0] and then re-estimate k=y[0]+1/11*x[0,0]-7*x[1,0]. 
In this way, instead of {\color{blue}3.2794165} we get  3.279416504354730, which commonly improved the accuracy of the result to 16 digits.   
	
Since big data sets are estimated by functions with a very small number of parameters, chances for over-fitting are extremely low; therefore in most cases, we used all data as train sets. 


More detailed explanations are given in the Report.
