"""
Quickstart: f-Greedy Optimal Recovery with Mode Selection
---------------------------------------------------------
This script runs the f-Greedy variant of the optimal recovery method 
for a nonlinear PDE using Gaussian kernels. It supports two nonlinearities:
  - "poly": τ(u) = u³
  - "sin" : τ(u) = (1/20) * sin(π u)

Workflow:
  1. Select nonlinearity mode via `MODE` ("poly" or "sin")
  2. Initialize kernel and solution function
  3. Run a short greedy_loop to pre-compile Numba functions
  4. Run `greedy_loop` for the specified number of iterations
  5. Record:
       - Max solution error per iteration
       - PDE residual per iteration
       - Boundary residual per iteration
  6. Save results to CSV with timestamp
  7. Plot log-scale curves of errors, residuals, and boundary residuals

Results:
  - CSV file: results/greedy_loop_output_<timestamp>.csv
  - Log-scale plots of error evolution

Configurable parameters:
  - gamma (kernel width, default: 10)
  - iterations (number of f-Greedy steps, default: 500)
  - MODE ("poly" or "sin") for nonlinearity
  - solution_function ("kernel", "sin", "c2")
"""

import numpy as np
import time
from numba import njit
import pandas as pd
import numpy as np
from datetime import datetime
from assets.Gaussian_Kernel_JIT import Kernel as KernelClass
from assets.collocation_functions import greedy_loop
import matplotlib.pyplot as plt

# --- Settings ---
gamma = 10
Kernel = KernelClass(gamma)
iterations = 500
MODE = "poly"  # Change to "sin" for sine version
solution_function ='kernel' #choose between kernel, sin and c2
# --- Nonlinearity ---
@njit
def tau_poly(x):
    return x ** 3

@njit
def tau_poly_der(x):
    return 3 * x ** 2

@njit
def tau_poly_2der(x):
    return 6 * x

# -----------------------------
# Sin version: tau = 1/20 sin(pi x)
# -----------------------------
@njit
def tau_sin(x):
    return (1 / 20) * np.sin(np.pi * x)

@njit
def tau_sin_der(x):
    return (1 / 20) * np.pi * np.cos(np.pi * x)

@njit
def tau_sin_2der(x):
    return - (1 / 20) * np.pi ** 2 * np.sin(np.pi * x)

# -----------------------------
# Select implementation here
# -----------------------------


if MODE == "poly":
    tau = tau_poly
    tau_der = tau_poly_der
    tau_2der = tau_poly_2der
elif MODE == "sin":
    tau = tau_sin
    tau_der = tau_sin_der
    tau_2der = tau_sin_2der
else:
    raise ValueError("Unknown MODE: must be 'poly' or 'sin'")

# --- Solutions ---
def f_sin(points):
    x = np.pi * points[:, 0, np.newaxis]
    y = np.pi * points[:, 1, np.newaxis]
    return np.sin(x) * np.sin(y)
def laplace_f_sin(points):
    return -2 * np.pi**2 * f_sin(points)
def f_c2(points):
    dx = points[:, 0:1] - 0.5
    dy = points[:, 1:2] - 0.5
    r2 = dx**2 + dy**2
    return r2**2.5
def laplace_f_c2(points):
    dx = points[:, 0:1] - 0.5
    dy = points[:, 1:2] - 0.5
    r2 = dx**2 + dy**2
    return 25 * r2**1.5
kernel_for_sol=KernelClass(10)
f_k = lambda x: kernel_for_sol.kernel_matrix(x, np.array([[0.2, 0.5]]))
laplace_f_k = lambda x: kernel_for_sol.laplace_on_x_matrix(x, np.array([[0.2, 0.5]]))


# --- solution choice logic ---
function_dict = {
    "kernel": (f_k, laplace_f_k),
    "sin": (f_sin, laplace_f_sin),
    "c2": (f_c2, laplace_f_c2)
}

f, laplace_f = function_dict[solution_function]

rhs = lambda x: laplace_f(x) + tau(f(x))
boundary_condition = f

#run to force numba compile for more accurate time measurment
interpolant, coeff, points, boundary_points, residuals, boundary_residuals, errors, failed_early, times, _ = greedy_loop( 
    tau, 
    tau_der, 
    rhs, 
    boundary_condition, 
    3, 
    Kernel, 
    points = np.array([]), 
    boundary_points = np.array([]), 
    sol = f,
    verbosity = True,
    eps = 10 **- 10)

start_time = time.time()
interpolant, coeff, points, boundary_points, residuals, boundary_residuals, errors, failed_early, times, _ = greedy_loop( 
    tau, 
    tau_der, 
    rhs, 
    boundary_condition, 
    iterations, 
    Kernel, 
    points = np.array([]), 
    boundary_points = np.array([]), 
    sol = f,
    verbosity = True,
    eps = 10 **- 10)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.4f} seconds")

# Get current date and time string
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create DataFrame
data = {
    'Index': np.arange(len(errors)),
    'Error': errors,
    'Residual': residuals,
    'BoundaryResidual': boundary_residuals
}
df = pd.DataFrame(data)

# Save to results/ folder with timestamp
filename = f"results/greedy_loop_output_{timestamp}.csv"
df.to_csv(filename, index=False)

print(f"Saved results to: {filename}")

import matplotlib.pyplot as plt
import numpy as np

# Assuming these are already computed by greedy_loop
# errors, residuals, boundary_residuals = ...

indices = np.arange(len(errors))

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Error plot
axs[0].semilogy(indices, errors, marker='o')
axs[0].set_title('Error (log scale)')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Value')
axs[0].grid(True, which='both', linestyle='--')

# Residual plot
axs[1].semilogy(indices, residuals, marker='o', color='orange')
axs[1].set_title('Residual (log scale)')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Value')
axs[1].grid(True, which='both', linestyle='--')

# Boundary residual plot
axs[2].semilogy(indices, boundary_residuals, marker='o', color='green')
axs[2].set_title('Boundary Residual (log scale)')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Value')
axs[2].grid(True, which='both', linestyle='--')

plt.tight_layout()
plt.show()
