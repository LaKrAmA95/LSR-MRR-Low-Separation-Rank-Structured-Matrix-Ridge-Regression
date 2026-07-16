# This code was implemented by Lakshitha Ramanayake.
# The function wrappper for the LSR Structured Ridge Regression for Matrix covariates.
# The wrapper is for HCP to find out whether improving the Lambda would give good results.


#importing essential libraries 
import os 
import sys
import subprocess
import time 

#installing the libraries
def install_packages():
    subprocess.check_call([sys.executable,"-m","pip","install","-r","requirement.txt"]) 
    
#import the installed libraries 

import datetime
import re
import dill
import platform
import copy


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy

from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler


#importing coded functions form the directory
from KFoldCV import KFoldCV
from train_test_los import train_test
from Load_data import load_data
from Load_data import samplestomat
from Load_data import vectomat_matlab
from Load_data import normalize_by_frobenius_norm

#initializer
from LSR_Tensor_2D_v1 import LSR_tensor_dot

#loading data 
import dill
import pickle

# Changing the directory to the current directory.

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

################################################################################ Loading Initializer

load_initializer = False 

################################################################################1. Importing Data  

#Reading the needed files 
alphas = [0.5,5,8,9,10,11,12,13,14,15,16,17,18,19,20]
data_directory = os.path.join(current_directory,'Data')
seed = 35

#traning and testing splits 
fmri_rs,df_train, df_test = load_data(seed,data_directory)

#Create train and test arrays
train_subjects = df_train.index.to_list()
test_subjects = df_test.index.to_list()

#Reshape labels into column vector
X_train_vec, Y_train = fmri_rs[train_subjects], df_train["varimax_cog"].to_numpy().reshape((-1, 1))
X_test_vec, Y_test = fmri_rs[test_subjects], df_test["varimax_cog"].to_numpy().reshape((-1, 1))

################################################################################2. Preprocessing 

#Training data
X_train = samplestomat(X_train_vec,400)
X_test  = samplestomat(X_test_vec,400)
Y_train = Y_train.reshape(-1)
Y_test = Y_test.reshape(-1)


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#Function to row wise normalization

X_train = normalize_by_frobenius_norm(X_train)
X_test = normalize_by_frobenius_norm(X_test)

#number of samples in training and testing 

n_train = X_train.shape[0]
n_test = X_test.shape[0]


# Reshape the 3D array to a 2D array where each row represents a sample
# The shape of the original 3D array is (n_samples, n_features_per_sample, n_dimensions)
# We reshape it to (n_samples, n_features_per_sample * n_dimensions)

X_train_2D = X_train.reshape(n_train, -1)
X_test_2D = X_test.reshape(n_test,-1)

# Initialize StandardScaler
scaler = StandardScaler(with_std = False) #standard scalar only

# Fit scaler on train data and transform train data
X_train_scaled = scaler.fit_transform(X_train_2D)
# Transform test data using the scaler fitted on train data
X_test_scaled = scaler.transform(X_test_2D)

# Reshape the scaled data back to 3D
X_train = X_train_scaled.reshape(n_train, X_train.shape[1],X_train.shape[2])
X_test  = X_test_scaled.reshape(n_test, X_test.shape[1],X_train.shape[2])

#average response value
Y_train_mean = np.mean(Y_train)
# Mean centering y_train and y_test
Y_train = Y_train - Y_train_mean


print("Sample mean for each feature (across samples):",scaler.mean_)
print("Sample variance for each feature (across samples):",scaler.var_)
print('Response Average:',Y_train_mean)

################################################################################3. Setting the hyper parameters

#Tensor Decomposition
tensor_dimensions = np.array([X_train.shape[1], X_train.shape[2]])
tensor_mode_ranks = np.array([8,8])
separation_rank = 2


#For now, define finite alpha set that we are searching over


#algorithm 
max_iter = 30
threshold = 1e-4

#setting up hyper parameters
hypers = {'max_iter': max_iter, 'threshold': threshold, 'ranks': tuple(tensor_mode_ranks), 'separation_rank': separation_rank}

#setting up parameters for the initializer
ranks = hypers['ranks']
separation_rank = hypers['separation_rank']
LSR_tensor_dot_shape = tuple(X_train.shape)[1:]
need_intercept = False 

realizations = 10

for real in range(realizations):
    
    # initializer

    if load_initializer is True:
        #loading file path
        loading_path = os.path.join(current_directory, 'Experimental_Results','Sep_2_Tucker_44','Lambda_Over_Search')
        if platform.system() == "Windows":
            pkl_file = rf""
        elif platform.system() == "Darwin":
            pkl_file = f""
        elif platform.system() == "Linux":
            pkl_file = os.path.join(loading_path,f"9_Seed_35_Lambda_10_n_train_676_n_test_77_tensor_mode_ranks_[4 4]_separation_rank_2.pkl")
            
        file= open(pkl_file, 'rb')
        data = pickle.load(file)
        file.close()
        
        lsr_tensor_init = data[-1]
    else:  
        lsr_tensor_init = LSR_tensor_dot(shape=LSR_tensor_dot_shape, ranks=ranks, separation_rank=separation_rank, intercept=need_intercept)

    lsr_tensor_to_save = copy.deepcopy(lsr_tensor_init)
    
    for lambda1 in alphas:
        
        ################################################################################
        # 5. Taking Oracle Results
        start_time = time.time()
        lsr_ten_lambda = copy.deepcopy(lsr_tensor_init)
        normalized_estimation_error, test_nmse_loss, test_R2_loss, test_correlation, gradient_values, factor_core_iteration = \
            train_test(X_train, Y_train, X_test, Y_test, lambda1, hypers, Y_train_mean, lsr_tensor=lsr_ten_lambda, B_tensored=None, intercept=False)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Time for one lambda: {execution_time}')

        ################################################################################
        # 6. Storing 
        storing_directory = os.path.join(current_directory, 'Experimental_Results', 'Sep_2_Tucker_88', 'Lambda_Over_Search')

        if platform.system() == "Windows":
            pkl_file = os.path.join(storing_directory,
                                    f"{real}_Seed_{seed}_Lambda_{lambda1}_n_train_{n_train}_n_test_{n_test}_tensor_mode_ranks_{tensor_mode_ranks}_separation_rank_{separation_rank}.pkl")
        elif platform.system() == "Darwin":
            pkl_file = os.path.join(storing_directory,
                                    f"{real}_Seed_{seed}_Lambda_{lambda1}_n_train_{n_train}_n_test_{n_test}_tensor_mode_ranks_{tensor_mode_ranks}_separation_rank_{separation_rank}.pkl")
        elif platform.system() == "Linux":
            pkl_file = os.path.join(storing_directory,
                                    f"{real}_Seed_{seed}_Lambda_{lambda1}_n_train_{n_train}_n_test_{n_test}_tensor_mode_ranks_{tensor_mode_ranks}_separation_rank_{separation_rank}.pkl")
        else:
            raise RuntimeError("Unsupported operating system")

        with open(pkl_file, "wb") as file:
            dill.dump((X_train, Y_train, X_test, Y_test, Y_train_mean, lambda1, normalized_estimation_error, 
                       test_nmse_loss, test_R2_loss, test_correlation, gradient_values, factor_core_iteration, lsr_tensor_to_save), file)
