#Import necessary libraries for the assignment
#from IPython.display import display, Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import skimage as skimage
from skimage import data, io, filters
import matplotlib.cm as cm
import math

# Create matrix A

A = [[0, 1, 1], [math.sqrt(2), 2, 0], [0, 1, 1]]


A = np.array(A)


# Transpose the matrix

A_t = A.transpose()


# Multiply A with A_t
AA = np.dot(A_t, A)


# Create lambda matrix
from sympy import symbols, solve

# Create a variable l which stands for lambda

l = symbols('l')

lambda_matrix = [[l, 0, 0], [0, l, 0], [0, 0, l]]

lambda_matrix = np.array(lambda_matrix)


# Solving the equation det(AA - lambda*I) = 0
det_matrix = AA-lambda_matrix

equation = det_matrix[0][0]*(det_matrix[1][1]*det_matrix[2][2]-det_matrix[2][1]*det_matrix[1][2])-det_matrix[1][0]*(det_matrix[0][1]*det_matrix[2][2]-det_matrix[2][1]*det_matrix[0][2])

#Solve equation = 0

result = solve(equation)

print('Lambda values are: ', result)

# The singular values are the square root of the eigenvalues:

print('Singular values: ')
for i in result:
    print(math.sqrt(i))