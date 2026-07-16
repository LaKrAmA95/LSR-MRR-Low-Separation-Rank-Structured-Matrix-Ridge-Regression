#Importing Needed Libraries 
import os
import pickle
import platform 
from _utils.optimization import inner_product,R2
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

number = 10

# Changing the directory to the current directory.
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)


#defining the data directory
data_directory = os.path.join(current_directory,'Experimental_Results','Sep_2_Tucker_88','Lambda_Over_Search')

for i in range(number):
    # Get all .pkl files in the folder
    pkl_files = [file for file in os.listdir(data_directory) if file.startswith(str(i)) and file.endswith('.pkl')]

    # Initialize an empty DataFrame to store results for multiple runs
    results_df = pd.DataFrame(columns=[
        'Lambda','Train NMSE Loss','Test NMSE Loss', 'Train R2 Score', 'Test R2 Score','Train Correlation','Test Correlation'
    ])

    for pkl_file in pkl_files:
        file_path = os.path.join(data_directory ,pkl_file)
        with open(file_path,'rb') as f:
            
            #seed extracting segment 
            
            # Get the second element divided by _ which is '11'
            seed_part = pkl_file.split('_')[2]
            seed_number = int(seed_part)
            print(seed_number)
            
            data = pickle.load(f)
            
            X_train = data[0]
            Y_train = data[1]
            X_test  = data[2]
            Y_test  = data[3]
            Y_train_mean = data[4]
            lambda1 = data[5]
            test_normalized_estimation_error = data[6] #11
            test_nmse_loss = data[7] #12
            test_R2_loss = data[8] #13
            test_correlation = data[9] #14
            factor_core_iterates = data[11] #16
            
            #Model
            expanded_lsr = factor_core_iterates[-1].expand_to_tensor()
            expanded_lsr = np.reshape(expanded_lsr, X_train[0].shape, order = 'F')    
        
            #Training_Error_Regeneration
            Y_train_predicted = inner_product(np.transpose(X_train, (0, 2, 1)), expanded_lsr.flatten(order ='F')) + factor_core_iterates[-1].b 
            
            #Training Errors
            
            train_nmse_loss = np.sum(np.square((Y_train_predicted.flatten() - Y_train.flatten()))) / np.sum(np.square(Y_train.flatten()))
            train_R2_loss = R2(Y_train.flatten(), Y_train_predicted.flatten())
            train_correlation = np.corrcoef(Y_train_predicted.flatten(), Y_train.flatten())[0, 1]
        
            #Testing Errors
            
            Y_test_predicted = inner_product(np.transpose(X_test, (0, 2, 1)), expanded_lsr.flatten(order ='F')) + factor_core_iterates[-1].b + Y_train_mean

            test_nmse_loss = np.sum(np.square((Y_test_predicted.flatten() - Y_test.flatten()))) / np.sum(np.square(Y_test.flatten()))
            test_R2_loss = R2(Y_test.flatten(), Y_test_predicted.flatten())
            test_correlation = np.corrcoef(Y_test_predicted.flatten(), Y_test.flatten())[0, 1]
            
            # Append the results of this run to the DataFrame
            
                # Create a new DataFrame for this run
            new_row = pd.DataFrame({
            'Lambda': [lambda1],
            'Train NMSE Loss': [train_nmse_loss],
            'Test NMSE Loss': [test_nmse_loss],
            'Train R2 Score': [train_R2_loss],
            'Test R2 Score': [test_R2_loss],
            'Train Correlation': [train_correlation],
            'Test Correlation': [test_correlation]
            })
            
            # Concatenate the new row to the results DataFrame
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
    #sorting the dataframe based on seed 
    results_df  = results_df.sort_values(by='Lambda')

    # Reset the index (optional)
    results_df.reset_index(drop=True, inplace=True)


    # Save the DataFrame as a CSV file
    results_df.to_csv(f'Lambda_Over_Search_Sep_2_{i}_Tuck_88.csv', index=False)  # Set index=False to avoid writing row numbers
    
