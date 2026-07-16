# This code was implemented by Lakshitha Ramanayake.
# The functionn wrappper for the LSR Structured Ridge Regression for Matrix covariates.
# The wrapper is for HCP data.

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
import multiprocessing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy

from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler


#importing coded functions form the directory
from KFoldCV import KFoldCV
from train_test import train_test
from Load_data import load_data
from Load_data import samplestomat
from Load_data import vectomat_matlab
from Load_data import normalize_by_frobenius_norm
from _utils.LSR_Tensor_2D_v1 import LSR_tensor_dot


# This wrapping is done to prevent the code from being executed all over again in the sub-processes.
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Use 'spawn' instead of 'fork'
    
    # Changing the directory to the current directory.
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)


    ################################################################################0. Setting up all the parameters
    
    #Data related variables
    seed = 47    
    # Tensor decomposition hyper-parameters
    tensor_mode_ranks = np.array([4,4])
    separation_rank = 3 
    
    #cross validation 
    k_folds = 10
    
    #For now, define finite ridge regression parameters
    alphas = [10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5]


    #count of initializers
    num_init = 10	
    
    #Optimization Algorithm  
    max_iter = 30
    threshold = 1e-5
    
    ################################################################################1. Importing Data  

    #Reading the needed files 
    data_directory = os.path.join(current_directory,'Data')
    
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
    LSR_tensor_dot_shape = tuple(X_train.shape)[1:]

    #initializers 
    base_tensors = [LSR_tensor_dot(shape=LSR_tensor_dot_shape, ranks=tensor_mode_ranks, separation_rank=separation_rank, intercept=False) for _ in range(num_init)]

    ################################################################################4. Training 

    hypers = {'max_iter': max_iter, 'threshold': threshold, 'ranks': tuple(tensor_mode_ranks), 'separation_rank': separation_rank}

    #  function to evaluate parallel processes
    #  KF Cross-Validation for alpha will be run on each initializer and the best accuracy over different initializers will be checked.
    
    def run_processes():
            # Start time calculation
            start_time = time.perf_counter()

            # Initialize the queue
            output_queue = multiprocessing.Queue()
            processes = []

            # Launch processes for each base tensor
            for token,base_tensor in enumerate (base_tensors):
                p = multiprocessing.Process(
                    target=KFoldCV, 
                    args=[X_train, Y_train, alphas, k_folds, hypers, base_tensor, output_queue,token,seed], 
                    kwargs={'B_tensored': None, 'intercept': False}
                )
                p.start()
                processes.append(p)

            # Ensure all processes finish
            for process in processes:
                process.join()

            # Collect results from the queue
            all_results = []
            while not output_queue.empty():
                try:
                    # Add timeout to prevent infinite wait
                    result = output_queue.get(timeout=5)  
                    all_results.append(result)
                except multiprocessing.queues.Empty:
                    print("Queue is empty, but some processes might not have put results.")

            # Finish time calculation
            finish_time = time.perf_counter()
            execution_time = finish_time - start_time
            print(f'Execution Time: {execution_time:.2f} second (s).')         
            return all_results
         
    #executing the parallel process
    best_lambda_result_for_initializer = run_processes()
    
    #printing 
    print(best_lambda_result_for_initializer)
    
    
    # Convert to numpy array
    best_lambda_each_init = np.array(best_lambda_result_for_initializer)
    # Use argmax to find the index of the maximum value in validation correlation 
    max_index = np.nanargmax(best_lambda_each_init[:, 2]) 
        
    #Best Parameters 
    initializer = base_tensors[max_index]
    lambda1 = best_lambda_each_init[max_index,1]
    
    
    
    ################################################################################5. Testing 

    start_time = time.time()

    hypers = {'max_iter': max_iter, 'threshold': threshold, 'ranks': tuple(tensor_mode_ranks), 'separation_rank': separation_rank}
    normalized_estimation_error, test_nmse_loss, test_R2_loss, test_correlation,gradient_values,factor_core_iteration\
         = train_test(X_train, Y_train, X_test, Y_test, lambda1, hypers,Y_train_mean,initializer,B_tensored = None, intercept= False)

    end_time = time.time()

    print(f'Error Metric: NMSE:{test_nmse_loss}, R2:{test_R2_loss}, CORR:{test_correlation} ')
    execution_time = end_time - start_time
    print(f'Time for one lambda:{execution_time}')


    ################################################################################6. Storing 

    storing_directory = os.path.join(current_directory,'Experimental_Results','Sep_3_Tucker_44_0')


    if platform.system() == "Windows":
        pkl_file = os.path.join(storing_directory,\
         f"Seed_{seed}_n_train_{n_train}_n_test_{n_test}_tensor_mode_ranks_{tensor_mode_ranks}_separation_rank_{separation_rank}.pkl")
    elif platform.system() == "Darwin":
        pkl_file = os.path.join(storing_directory,\
         f"Seed_{seed}_n_train_{n_train}_n_test_{n_test}_tensor_mode_ranks_{tensor_mode_ranks}_separation_rank_{separation_rank}.pkl")
    elif platform.system() == "Linux":
        pkl_file = os.path.join(storing_directory,\
         f"Seed_{seed}_n_train_{n_train}_n_test_{n_test}_tensor_mode_ranks_{tensor_mode_ranks}_separation_rank_{separation_rank}.pkl")
    else:
        raise RuntimeError("Unsupported operating system")


    with open(pkl_file, "wb") as file:
        dill.dump((X_train, Y_train, X_test, Y_test,Y_train_mean,lambda1,normalized_estimation_error, test_nmse_loss,\
                test_R2_loss, test_correlation,gradient_values,factor_core_iteration,initializer), file)
