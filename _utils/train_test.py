from sklearn.linear_model import Ridge
import numpy as np
from LSR_Tensor_2D_v1 import LSR_tensor_dot
from SLICERR import SLICERR
from optimization import inner_product, R2, objective_function_vectorized
import copy

def train_test(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, lambda1, hypers,Y_train_mean,initializer: object,B_tensored = None, intercept = False):
  hypers['weight_decay'] = lambda1

  #Define LSR Tensor Hyperparameters
  ranks = hypers['ranks']
  separation_rank = hypers['separation_rank']
  LSR_tensor_dot_shape = tuple(X_train.shape)[1:]
  need_intercept = intercept

  #Construct LSR Tensor
  lsr_tensor_before  = copy.deepcopy(initializer)
  try:
    lsr_tensor,gradient_values,factor_core_iteration = SLICERR(lsr_tensor_before, X_train, Y_train, hypers,intercept = need_intercept)
    
    expanded_lsr = lsr_tensor.expand_to_tensor()
    expanded_lsr = np.reshape(expanded_lsr, X_train[0].shape, order = 'F')

    Y_test_predicted = inner_product(np.transpose(X_test, (0, 2, 1)), expanded_lsr.flatten(order ='F')) + lsr_tensor.b + Y_train_mean

    print(f"Y_test_predicted: {Y_test_predicted.flatten()}, Y_test: {Y_test.flatten()}")
    
    test_nmse_loss = np.sum(np.square((Y_test_predicted.flatten() - Y_test.flatten()))) / np.sum(np.square(Y_test.flatten()))
    if B_tensored is not None: normalized_estimation_error = ((np.linalg.norm(expanded_lsr - B_tensored)) ** 2) /  ((np.linalg.norm(B_tensored)) ** 2)
    test_R2_loss = R2(Y_test.flatten(), Y_test_predicted.flatten())
    test_correlation = np.corrcoef(Y_test_predicted.flatten(), Y_test.flatten())[0, 1]
  
  except Exception as e:
    print('An Error encountered while Testing0')
    normalized_estimation_error = np.nan
    test_nmse_loss = np.nan 
    test_correlation = np.nan
    test_R2_loss = np.nan
    gradient_values = np.nan
    factor_core_iteration = np.nan


  if B_tensored is not None:
    return normalized_estimation_error, test_nmse_loss, test_R2_loss, test_correlation,gradient_values,factor_core_iteration
  else:
    normalized_estimation_error = np.inf
    return normalized_estimation_error, test_nmse_loss, test_R2_loss, test_correlation,gradient_values,factor_core_iteration
