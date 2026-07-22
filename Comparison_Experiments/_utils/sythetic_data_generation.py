import numpy as np 
import random


def generate_data(n_data,M,P,sep,noise_var,intercept=0,random_seed=42, verbose=False):
    
    '''
    This code generates data (matrix covariate,scalar_response) for linear regression where the underlying coefficient matrix adheres
    to an LSR structure.

    Input:
    n_data: number of data points 
    M: data dimensions
    P: mode ranks of the matrix
    sep: separation rank 
    noise_var: noise variance of the linear model 
    intercept: intercept of the linear model 
    random_seed: seed for reproducability

    Output:
    X: samples 
    y: responses
    coeff: coefficient tensor 
    factor_matrices: list containing factor matrices 
    core_mat: core tensor
    y_no_noise: responses before noise addition 

    '''

    # fixing the random number generator 
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ******************* CORE MATRIX GENERATION *******************
    g_length = np.prod(P)
    g = np.random.normal(0,1,g_length)
    core_mat = g.reshape(P,order='F') 

    # ******************* FACTOR MATRICES GENERATION *******************
    factor_matrices = []
    for s in range(sep):
        summand_factor = []
        for k in range(len(M)):
            fac_k_s = np.random.normal(0,1,size = (M[k],P[k]))
            summand_factor.append(fac_k_s)
        factor_matrices.append(summand_factor)
    
    if verbose:
        print('Factor Matrices and Core Tensor Initialized')

    # ******************* COEFFICIENT MATIX *******************
    coeff = np.zeros([*M])
    for s in range(sep):
        coeff += factor_matrices[s][0]@ core_mat @ factor_matrices[s][1].T

    # ******************* SYNTHETIC DATA GENERATION *******************
    
    #X
    data_cude_dim = [n_data,M[0],M[1]]
    X = np.random.normal(0,1,size=data_cude_dim)
    X_tr = np.transpose(X,(0,2,1))
    X_tr_vec = np.reshape(X_tr, newshape = (n_data,-1))


    #y
    y_no_noise = X_tr_vec @ coeff.flatten(order='F') + intercept
    y = y_no_noise + np.random.normal(0,np.sqrt(noise_var),size = n_data)

    if verbose:
        print('Synthetic Data Generation Competed.')
    
    if verbose and intercept != 0:
        print('Data Generated with and Intercept.')
    
    return X,y,coeff,factor_matrices,core_mat,y_no_noise


def generate_data_3D(
    n_data,
    M,
    P,
    sep,
    noise_var,
    intercept=0,
    random_seed=42,
    verbose=False,
):
    """
    Generate third-order tensor covariates and scalar responses for LSR
    regression.

    The coefficient tensor is

        coeff = sum_s G x1 A[s, 0] x2 A[s, 1] x3 A[s, 2],

    where G is a shared core tensor with shape P and A[s, k] has shape
    (M[k], P[k]). Responses follow

        y[i] = <X[i], coeff> + intercept + noise[i],

    with noise variance ``noise_var``.

    Returns
    -------
    X, y, coeff, factor_matrices, core_tensor, y_no_noise
    """
    if not isinstance(n_data, (int, np.integer)) or n_data <= 0:
        raise ValueError("n_data must be a positive integer.")
    if len(M) != 3 or len(P) != 3:
        raise ValueError("M and P must each contain exactly three dimensions.")
    if any(not isinstance(dim, (int, np.integer)) or dim <= 0 for dim in M):
        raise ValueError("All dimensions in M must be positive integers.")
    if any(not isinstance(rank, (int, np.integer)) or rank <= 0 for rank in P):
        raise ValueError("All ranks in P must be positive integers.")
    if not isinstance(sep, (int, np.integer)) or sep <= 0:
        raise ValueError("sep must be a positive integer.")
    if noise_var < 0:
        raise ValueError("noise_var must be nonnegative.")

    rng = np.random.default_rng(random_seed)

    core_tensor = rng.normal(0, 1, size=tuple(P))

    factor_matrices = []
    for _ in range(sep):
        summand_factors = [
            rng.normal(0, 1, size=(M[k], P[k])) for k in range(3)
        ]
        factor_matrices.append(summand_factors)

    if verbose:
        print("Factor matrices and core tensor initialized.")

    coeff = np.zeros(tuple(M))
    for factor_1, factor_2, factor_3 in factor_matrices:
        coeff += np.einsum(
            "abc,ia,jb,kc->ijk",
            core_tensor,
            factor_1,
            factor_2,
            factor_3,
            optimize=True,
        )

    X = rng.normal(0, 1, size=(n_data, *M))
    y_no_noise = np.einsum("nijk,ijk->n", X, coeff, optimize=True) + intercept
    y = y_no_noise + rng.normal(0, np.sqrt(noise_var), size=n_data)

    if verbose:
        print("Synthetic 3D data generation completed.")
        if intercept != 0:
            print("Data generated with an intercept.")

    return X, y, coeff, factor_matrices, core_tensor, y_no_noise
