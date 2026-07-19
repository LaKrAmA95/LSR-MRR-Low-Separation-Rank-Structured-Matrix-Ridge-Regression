from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import numpy as np
import copy
from LSR_Tensor_2D_v1 import LSR_tensor_dot
from SLICERR import SLICERR
from optimization import inner_product, R2, objective_function_vectorized
import multiprocessing 
import os 

def KFoldCV(X_train: np.ndarray, Y_train: np.ndarray, alphas, k_folds, hypers,base_tensor,output_queue,token,seed,B_tensored = None, intercept = False):
  
  kfold = KFold(n_splits=k_folds, shuffle=True)

  #Matrix storing Validation Results
  if B_tensored is not None:
    validation_normalized_estimation_error = np.ones(shape = (k_folds, len(alphas))) * np.inf
  validation_nmse_losses = np.ones(shape = (k_folds, len(alphas)))* np.inf
  validation_correlations = np.zeros(shape = (k_folds, len(alphas)))
  validation_R2_scores = np.zeros(shape = (k_folds, len(alphas)))

  #Define LSR Tensor Hyperparameters
  ranks = hypers['ranks']
  separation_rank = hypers['separation_rank']
  need_intercept = intercept

  # Repeating the same initializer for all the folds.
  # Create independent copies for each fold and alpha.
  lsr_tensors = [[copy.deepcopy(base_tensor) for _ in range(k_folds)] for _ in range(len(alphas))]

  # Array to hold gradient values 
  gradient_information = np.ones(shape = (k_folds, len(alphas), hypers['max_iter'], separation_rank, len(ranks) + 1))
  
  #Go thru each fold
  #to handle errors
  for fold, (train_ids, validation_ids) in enumerate(kfold.split(X_train)):
      X_train_updated, Y_train_updated = X_train[train_ids], Y_train[train_ids]
      X_validation, Y_validation = X_train[validation_ids], Y_train[validation_ids]

      for index1, alpha1 in enumerate(alphas):
        
        try:
          
          hypers['weight_decay'] = alpha1

          lsr_ten,gradient_values,factor_core_iteration = SLICERR(lsr_tensors[index1][fold], X_train_updated, Y_train_updated, hypers, intercept = need_intercept)
          
          #validation 
          expanded_lsr = lsr_ten.expand_to_tensor()
          expanded_lsr = np.reshape(expanded_lsr, X_validation[0].shape, order='F')
          Y_validation_predicted = inner_product(np.transpose(X_validation, (0, 2, 1)), expanded_lsr.flatten(order ='F')) + lsr_ten.b
      
          #error matrices 
          if B_tensored is not None:  
            normalized_estimation_error = ((np.linalg.norm(expanded_lsr - B_tensored)) ** 2) /  ((np.linalg.norm(B_tensored)) ** 2)
          validation_nmse_loss = np.sum(np.square((Y_validation_predicted.flatten() - Y_validation.flatten()))) / np.sum(np.square(Y_validation.flatten()))
          correlation = np.corrcoef(Y_validation_predicted.flatten(), Y_validation.flatten())[0, 1]
          R2_value = R2(Y_validation.flatten(), Y_validation_predicted.flatten())

          if B_tensored is not None:
            validation_normalized_estimation_error[fold, index1] = normalized_estimation_error   
          validation_nmse_losses[fold, index1] = validation_nmse_loss
          validation_correlations[fold, index1] = correlation
          validation_R2_scores[fold, index1] = R2_value

          #Store Objective Function Information
          gradient_information[fold, index1] = gradient_values 

          if B_tensored is not None:
            print(f"Initializer= {token}, Fold = {fold}, Alpha = {alpha1}, Normalized Estimation Error: {normalized_estimation_error}, NMSE: {validation_nmse_loss}, Correlation: {correlation}, R^2 Score: {R2_value}")
          else:
            print(f"Initializer= {token}, Fold = {fold}, Alpha = {alpha1}, NMSE: {validation_nmse_loss}, Correlation: {correlation}, R^2 Score: {R2_value}")

        except Exception as e:
            # Handle the error and continue with the next lambda value
            print(f"Initializer= {token}, Fold:{fold} = {fold} Lambda {alpha1}: Error occurred during cross-validation: {e}")
            continue
  
  #Average out validation results
  if B_tensored is not None:
    average_normalized_estimation_error = np.mean(validation_normalized_estimation_error, axis = 0)
  average_validation_nmse_losses = np.mean(validation_nmse_losses, axis = 0)
  average_validation_correlations = np.mean(validation_correlations, axis = 0)
  average_validation_R2_scores = np.mean(validation_R2_scores, axis = 0)

  # Changing the directory to the current directory.
  current_directory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(current_directory)
  storing_directory = os.path.join(current_directory,'Experimental_Results','Sep_3_Tucker_44_0','Validation_Results',f'Seed_{seed}',f'Int{token}_output.csv')
  # Save as a CSV file
  np.savetxt(storing_directory,average_validation_correlations.flatten(), delimiter=',', fmt='%s')
  
  #Get alpha value that performs the best
  flattened_avg_validation_correlation = average_validation_correlations.flatten()
  print(flattened_avg_validation_correlation)
  lambda1 = alphas[np.nanargmax(flattened_avg_validation_correlation)]
  
  if B_tensored is not None:
    output_queue.put((token,lambda1,flattened_avg_validation_correlation[np.nanargmax(flattened_avg_validation_correlation)]))
  else:
    validation_normalized_estimation_error = np.inf
    normalized_estimation_error = np.inf 
    output_queue.put((token,lambda1,flattened_avg_validation_correlation[np.nanargmax(flattened_avg_validation_correlation)]))
   
