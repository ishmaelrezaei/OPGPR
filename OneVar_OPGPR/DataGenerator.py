"""Online Parametric Gaussian Process Regression
    Author: Esmaeil Rezaei, 10/02/2022

Example: 2D toy example

Steps:
    Generate data
        - Generate an initial data set: X_initial, Y_initial
        - Generate a test data set: X_pred, Y_pred
        - Generate streaming data sets: X_i, Y_i where i = 1, 2, ...
    Initial training by consuming (X_initial, Y_initial)
    Testing by (X_pred, Y_pred)
    Update training when arriving a new data and then testing by (X_pred, Y_pred)

Therefore, run the following files respectively.
    1- 2D_DataGenerator.py
    2- 2D_Initial_Training.py
    3-
"""

# Import necessary libraries
import time
import os                    # Import the operating system module
import shutil                # Import the shutil to delete the directory
import pandas as pd          # Import the Pandas library for data manipulation
from pyDOE import lhs        # Import the Latin Hypercube Sampling (LHS) method from pyDOE
import autograd.numpy as np  # Import NumPy with autograd capabilities

# Set a random seed for reproducibility
np.random.seed(12345)




# Define data setup parameters
N = 10000                    # Number of data points
D = 1                        # Dimensionality of the data
lb = 0.0 * np.ones((1, D))   # Lower bound for input data
ub = 1.0 * np.ones((1, D))   # Upper bound for input data
noise = 0.2                  # Noise level in the data
subset_size = 30
N_initial = 200
N_prediction = 400

# Directory where you want to save the CSV file
directory = 'MyInitialData/'
# Check if the directory exists, and delete it if it does
if os.path.exists(directory):
    shutil.rmtree(directory)
    time.sleep(10) # Pause to let delete

# Create the directory
os.makedirs(directory)


# Directory where you want to save the CSV file
directory = 'MyOnlineData/'
# Check if the directory exists, and delete it if it does
if os.path.exists(directory):
    shutil.rmtree(directory)
    time.sleep(10) # Pause to let delete

# Create the directory
os.makedirs(directory)
os.makedirs(directory+'X/')
os.makedirs(directory+'Y/')

# Directory where you want to save the CSV file
directory = 'MyPredictionData/'
# Check if the directory exists, and delete it if it does
if os.path.exists(directory):
    shutil.rmtree(directory)
    time.sleep(10) # Pause to let delete

# Create the directory
os.makedirs(directory)


# Define 'f' function
def f(x):
    return x * np.sin(4 * np.pi * x)

# Generate a large data set for online learning part
X = lb + (ub - lb) * lhs(D, N)
Y = f(X) + noise * np.random.randn(N, 1)

# Get the shape of the data
(N, D) = X.shape

# Set a new random seed based on data size
np.random.seed(N)

# Shuffle the data
perm = np.random.permutation(N)
X = X[perm]
Y = Y[perm]


# Split the online learning data into smaller subsets and save them as CSV files
cut = int(np.floor(N/subset_size))
directory = 'MyOnlineData/'

for i in range(cut):
    X_i = X[subset_size*i:(i+1)*subset_size, :]
    Y_i = Y[subset_size*i:(i+1)*subset_size]
    filename = os.path.join(directory + 'X/', f'X_{i + 1}.csv')
    X_i = pd.DataFrame(X_i)
    X_i.to_csv(filename, index=False)
    filename = os.path.join(directory + 'Y/', f'Y_{i + 1}.csv')
    Y_i = pd.DataFrame(Y_i)
    Y_i.to_csv(filename, index=False)

# Generate a new set for initial data
directory = 'MyInitialData/'
X_i = lb + (ub - lb) * lhs(D, N_initial)
Y_i = f(X_i) + noise * np.random.randn(N_initial, 1)

# Save initial data as CSV files
filename = os.path.join(directory, f'X_initial.csv')
X_i = pd.DataFrame(X_i)
X_i.to_csv(filename, index=False)
filename = os.path.join(directory, f'Y_initial.csv')
Y_i = pd.DataFrame(Y_i)
Y_i.to_csv(filename, index=False)

# Generate a new set of predictive data
directory = 'MyPredictionData/'
X_i = lb + (ub - lb) * np.linspace(0, 1, N_prediction)[:, None]
Y_i = f(X_i)

# Save predictive data as CSV files
filename = os.path.join(directory, f'X_pred.csv')
X_i = pd.DataFrame(X_i)
X_i.to_csv(filename, index=False)
filename = os.path.join(directory, f'Y_pred.csv')
Y_i = pd.DataFrame(Y_i)
Y_i.to_csv(filename, index=False)
