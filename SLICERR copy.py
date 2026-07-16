# This code was implemented by Lakshitha Ramanayake (2023).

# importing _utils 
from _utils import LSR_Tensor_2D_v1
from _utils.optimization import objective_function_tensor_sep


# importing python libraries
import copy
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor


# SLCIERR:

def SLICERR(lsr_ten, training_data: np.ndarray, training_labels: np.ndarray, hypers: dict, intercept = False):
    
    """
    Fit a low-separation-rank tensor using block coordinate descent.

    The function alternately updates every factor matrix and the shared core
    tensor by solving ridge-regression subproblems. 

    Parameters
    ----------
    lsr_ten : LSR_Tensor_2D_v1
        Low-separation-rank tensor to optimize. The object supplies its shape,
        tensor ranks, separation rank, and order, together with the factor/core
        update, accessor, and design-matrix construction methods used by the
        block coordinate descent algorithm.
    training_data : numpy.ndarray
        Training design data "X". Its first dimension must equal the number
        of observations in "training_labels"; the remaining dimensions must
        be compatible with "lsr_ten".
    training_labels : numpy.ndarray
        Training targets "y", with one scalar response per observation.
    hypers : dict
        Optimization hyperparameters. Required keys are:
        - "weight_decay": nonnegative ridge regularization strength.
        - "max_iter": maximum number of block coordinate descent sweeps.
        - "threshold": stopping tolerance for the summed update norms.
    intercept : bool, default=False
        If "True", fit an intercept in each ridge subproblem and store the
        latest fitted intercept in "lsr_ten".

    Returns
    -------
    lsr_ten : LSR_Tensor_2D_v1
        The fitted tensor object. This is the same object supplied as input,
        with its factor matrices, core tensor, and optional intercept updated.
    factor_core_iteration : list of LSR_Tensor_2D_v1
        Deep copies of the fitted tensor state after each completed iteration,
        useful for inspecting or evaluating the optimization trajectory.
    """
    # get tensor factorization information
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order
    
    # get hyperparameters
    lambda1 = hypers["weight_decay"]
    max_iter = hypers["max_iter"]
    threshold = hypers["threshold"]
    
    # create models for each factor matrix and core matrix
    factor_matrix_models = [[Ridge(alpha = lambda1, solver = 'svd', fit_intercept = intercept) for _ in range(len(ranks))] for _ in range(sep_rank)]
    core_tensor_model = Ridge(alpha = lambda1, solver = 'svd', fit_intercept = intercept)
    
    # list to store iterate information 
    factor_core_iteration = []
    
    # run at most max_iter iterations of Block Coordinate Descent(BCD)
    
    for iteration in range(max_iter):
        
        # arrays to hod the iterate differences 
        factor_residuals = np.zeros(shape = (sep_rank, len(ranks)))
        core_residual = 0

        # array to store updates to factor matrices and core tensor
        updated_factor_matrices = np.empty((sep_rank, len(ranks)), dtype=object)
        updated_core_tensor = None

        #******************************** Factor Matrix Updates ********************************
        
        # iterate over factor matrices
        
        # separation rank loop
        for s in range(sep_rank):
            
            # tensor mode loop
            for k in range(len(ranks)):
                
                # Absorb Factor Matrices into X aside from (s, k) to get X_tilde

                X, y = training_data, training_labels
                # y tilde should now be y-b-<Q,X>_F
                X_tilde, y_tilde = lsr_ten.bcd_factor_update_x_y(s, k, X, y) 
                

                # solve the sub-problem pertaining to the factor tensor
                factor_matrix_models[s][k].fit(X_tilde, y_tilde)

                # retrieve original and updated factor matrices
                Bk = lsr_ten.get_factor_matrix(s, k)
                Bk1 = factor_matrix_models[s][k].coef_
                if intercept: b = factor_matrix_models[s][k].intercept_

                # shape Bk1 to the matrix 
                Bk1 = np.reshape(Bk1, (shape[k], ranks[k]), order = 'F')
            
                # update residuals and store updated factor matrix
                factor_residuals[s][k] = np.linalg.norm(Bk1 - Bk)
                updated_factor_matrices[s, k] = Bk1

                #Update Factor Matrix
                lsr_ten.update_factor_matrix(s, k, updated_factor_matrices[s, k])

                #update the intercept
                if intercept: lsr_ten.update_intercept(b)
                if intercept: b = lsr_ten.get_intercept()
                
            # end of iterating over the factor matrices
  

        #******************************** Core Tensor Update ********************************
        
        #absorb necessary matrices into X, aside from core tensor, to get X_tilde
        X, y = training_data, training_labels
        X_tilde, y_tilde = lsr_ten.bcd_core_update_x_y(X, y)

        #solve the sub-problem pertaining to the core tensor
        core_tensor_model.fit(X_tilde, y_tilde)

        #Get Original and Updated Core Tensor
        Gk = lsr_ten.get_core_matrix()
        Gk1 = np.reshape(core_tensor_model.coef_, ranks, order = 'F')
        if intercept: b = core_tensor_model.intercept_
        
        #Update Residuals and store updated Core Tensor
        core_residual = np.linalg.norm(Gk1 - Gk)
        updated_core_tensor = Gk1

        #Update Core Tensor
        lsr_ten.update_core_matrix(updated_core_tensor)

        #Update Intercept
        if intercept: lsr_ten.update_intercept(b)
        #Calculate Objective Function Value
        if intercept: b = lsr_ten.get_intercept()
        
        #saving lsr_ten
        factor_core_iteration.append(copy.deepcopy(lsr_ten))
        
        #Stopping Criteria
        diff = np.sum(factor_residuals.flatten()) + core_residual
        if diff < threshold: 
            print(rf'SLICERR converged after {iteration+1} iterations.')
            print('stopping_criterion_reached')
            break
            
    return lsr_ten,factor_core_iteration
