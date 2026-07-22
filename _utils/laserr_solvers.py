# This code is written by Lakshitha Ramanayake (2026).

# importing the libraries 
import numpy as np 
from .LSR_Tensor_2D_v1 import LSR_tensor_dot

def factor_matrix_update(X_tilde, y_tilde, lsr_ten, s,k, lambda_1):
   

   
   # the factor matrices 
   B_1 = lsr_ten.get_factor_matrix(s,0)
   B_2 = lsr_ten.get_factor_matrix(s,1)
   G = lsr_ten.get_core_matrix()

   # the rank s minus reconstruction of the LSR matrix 
   B_s = np.reshape(lsr_ten.expand_to_tensor(skip_term = s),lsr_ten.shape,order='F')

   
   if k == 0:
      
      I  = np.eye(B_1.shape[0])
      M  = (B_2 @ G.T).T @ (B_2 @ G.T)
      Q  = np.kron(M.T,I)
      c = np.kron(G,B_s) @ B_2.flatten(order = 'F')
      
   elif k == 1:
      
      I  = np.eye(B_2.shape[0])
      M  = (B_1 @ G).T @ (B_1 @ G) 
      Q  = np.kron(M,I)

      c = np.kron(G,B_s).T @ B_1.flatten(order = 'F')
   
   else:
      raise ValueError("Invalid value for k. Must be 1 or 2.")
   
   #coefficient matrix of the normal equations 
   coeff_matrix = X_tilde.T @ X_tilde + (lambda_1*Q)
   # solution vector for the normal euqations
   solution_vector = X_tilde.T @ y_tilde + lambda_1 * c
   
   # solving the normal equations 
   Bk1, residuals, rank, singular_values = np.linalg.lstsq(coeff_matrix,solution_vector,rcond=None)
  
   return Bk1


# This is the subroutine for the core tensor update.  
def core_tensor_update(X_tilde, y_tilde, lsr_ten, sep_rank, lambda_1):

   '''
   This function updates the core tensor sub-problem for the LASERR algorithm.
   For that, the subroutine solves the normal equations.
   
   Input:
   X_tilde: The absorbed training data matrix, after absorbing the factor matrices.
   y_tilde: The augmented training.
   sep_rank: The separation rank of the LSR decompostion.
   lsr_ten: The LSR tensor object. 
   lambda1: The regularization parameter.
   
   Output:
   Gk1: The core tenor update.

   '''

   # constructing the LSR structured matrix  (sum of kronecker structure)
   B = np.zeros((lsr_ten.get_factor_matrix(0,1).shape[0] * lsr_ten.get_factor_matrix(0,0).shape[0], lsr_ten.get_factor_matrix(0,1).shape[1] * lsr_ten.get_factor_matrix(0,0).shape[1]))
   for s in range(sep_rank):
        B += np.kron(lsr_ten.get_factor_matrix(s,1), lsr_ten.get_factor_matrix(s,0))



   # coefficient matrix of the normal equations 
   coeff_matrix = X_tilde.T @ X_tilde + (lambda_1 * B.T @ B )
   # solution vector for the normal equations
   solution_vector = X_tilde.T @ y_tilde

   Gk1, residuals, rank, singular_values = np.linalg.lstsq(coeff_matrix,solution_vector,rcond=None)
   
   return Gk1


