# SLICERR

SLICERR fits a matrix-valued linear regression model whose coefficient matrix
is represented by a low-separation-rank tensor decomposition. The optimizer
uses block coordinate descent: it alternately updates the factor matrices and
the core matrix by solving ridge-regression subproblems.

This README explains how to run `SLICERR` directly. It does not use the
cross-validation or hyperparameter-search pipelines included elsewhere in the
repository.

## Requirements

- Python 3.9 or newer
- NumPy
- scikit-learn

Install the Python dependencies from the repository root:

```powershell
python -m pip install numpy scikit-learn
```

## Input data

`SLICERR` expects the following inputs:

- `lsr_ten`: an initialized `LSR_tensor_dot` model.
- `training_data`: a NumPy array with shape `(n_samples, n_rows, n_columns)`.
- `training_labels`: a one-dimensional NumPy array with shape `(n_samples,)`.
- `hypers`: a dictionary containing `weight_decay`, `max_iter`, and
  `threshold`.
- `intercept`: whether an intercept should be fitted. The value used here
  should match the value used when constructing `LSR_tensor_dot`.

The model `shape` must equal `training_data.shape[1:]`. The `ranks` argument
contains the two Tucker ranks and must not exceed the corresponding dimensions
in `shape`.

## Minimal example

Create a file named `run_slicerr.py` in the repository root and add:

```python
import numpy as np

from SLICERR import SLICERR
from _utils.LSR_Tensor_2D_v1 import LSR_tensor_dot


# Make the random initialization reproducible.
np.random.seed(42)

# Generate example matrix-valued predictors.
n_samples = 100
n_rows = 8
n_columns = 6
X_train = np.random.normal(size=(n_samples, n_rows, n_columns))

# Generate scalar responses for the example.
true_coefficient = np.random.normal(size=(n_rows, n_columns))
y_train = np.einsum("nij,ij->n", X_train, true_coefficient)
y_train += 0.1 * np.random.normal(size=n_samples)

# Initialize the low-separation-rank coefficient model.
model = LSR_tensor_dot(
    shape=(n_rows, n_columns),
    ranks=(2, 2),
    separation_rank=1,
    intercept=True,
)

# Configure ridge regularization and block coordinate descent.
hypers = {
    "weight_decay": 0.1,
    "max_iter": 100,
    "threshold": 1e-6,
}

# Fit SLICERR. The input model is updated in place and also returned.
fitted_model, iteration_history = SLICERR(
    model,
    X_train,
    y_train,
    hypers,
    intercept=True,
)

# Generate predictions from the fitted coefficient matrix.
X_vectorized = np.transpose(X_train, (0, 2, 1)).reshape(n_samples, -1)
y_pred = X_vectorized @ fitted_model.expand_to_tensor()
y_pred += fitted_model.get_intercept()

training_mse = np.mean((y_train - y_pred) ** 2)
print(f"Completed iterations: {len(iteration_history)}")
print(f"Training MSE: {training_mse:.6f}")
```

Run the example from the repository root:

```powershell
python run_slicerr.py
```

## Hyperparameters

| Key | Meaning |
| --- | --- |
| `weight_decay` | Nonnegative ridge penalty applied to each subproblem. Larger values produce stronger regularization. |
| `max_iter` | Maximum number of complete block coordinate descent iterations. |
| `threshold` | Convergence tolerance. Training stops when the sum of factor and core update norms is below this value. |

## Outputs

`SLICERR` returns two values:

```python
fitted_model, iteration_history = SLICERR(...)
```

- `fitted_model` is the optimized `LSR_tensor_dot` object. It is the same
  object passed into the function and is modified in place.
- `iteration_history` is a list containing a deep copy of the model after each
  completed block coordinate descent iteration.

Useful fitted-model methods include:

- `fitted_model.expand_to_tensor()` returns the fitted coefficient matrix in
  column-wise vectorized form.
- `fitted_model.get_core_matrix()` returns the fitted core matrix.
- `fitted_model.get_factor_matrix(s, k)` returns factor matrix `k` from
  separation term `s`.
- `fitted_model.get_intercept()` returns the fitted intercept.

When the convergence tolerance is reached, SLICERR prints the iteration at
which it converged and stops before `max_iter`.
