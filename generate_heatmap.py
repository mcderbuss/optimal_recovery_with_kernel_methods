"""
Quickstart: f-Greedy Optimal Recovery
-------------------------------------
This script applies an f-Greedy variant of the optimal recovery method 
to solve a nonlinear PDE with Gaussian kernels. 

For each iteration, it records:
  - Max solution error (‖u - u_hat‖∞)
  - PDE residual error (‖Δu_hat + τ(u_hat) - RHS‖∞)
  - Boundary residual error (‖u_hat - f‖∞ on boundary)

Results:
  Saved to CSV: results/heatmap_output_<timestamp>.csv
  + Heatmap of absolute solution error
  + Heatmap of PDE residual with collocation points overlaid

Configurable parameters:
  - gamma (kernel width, default: 10)
  - iterations (number of f-Greedy steps, default: 500)
  - solution_function ("sin", "c2", "kernel")
"""


import numpy as np
import time
from numba import njit
from assets.Gaussian_Kernel_JIT import Kernel as KernelClass
from assets.collocation_functions import greedy_loop
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Settings ---
gamma = 10
Kernel = KernelClass(gamma)
iterations = 500
solution_function ='c2' #choose between kernel, sin and c2

# --- Nonlinearity ---
@njit
def tau(x):
    return x ** 3
@njit
def tau_der(x):
    return 3 * x ** 2
@njit
def tau_2der(x):
    return 6 * x


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

start_time = time.time()
interpolant, coeff, points, boundary_points, residuals, boundary_residuals, errors, failed_early, times, laplace_interpolant = greedy_loop( 
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
filename = f"results/heatmap_output_{timestamp}.csv"
df.to_csv(filename, index=False)

print(f"Saved results to: {filename}")

# --- Heatmap of error between true solution and interpolant ---

# Create a grid for evaluation
n_grid = 100
x = np.linspace(0, 1, n_grid)
y = np.linspace(0, 1, n_grid)
X, Y = np.meshgrid(x, y)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# Evaluate true solution and interpolant on the grid
true_vals = f(grid_points).flatten()
interp_vals = interpolant(grid_points).flatten()

# Compute error on grid
error_vals = np.abs(true_vals - interp_vals)
error_grid = error_vals.reshape((n_grid, n_grid))

# Plot the error heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_grid, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
ax.set_title('Heatmap of Error (True Solution - Interpolant)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')  # Force square plot area

# Overlay interpolation and boundary points
if len(points) > 0:
    ax.scatter(points[:, 0], points[:, 1], color='red', s=10, label='Interpolation Points')
if len(boundary_points) > 0:
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color='blue', s=10, label='Boundary Points')
ax.legend()

# Colorbar below
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.4)
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Absolute Error')

plt.tight_layout()
plt.show()


# --- Heatmap of PDE residual ---

# Reuse grid
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# Evaluate each term
interp_vals = interpolant(grid_points).flatten()
laplace_interp_vals = laplace_interpolant(grid_points).flatten()
nonlinear_term = interp_vals ** 3
rhs_vals = rhs(grid_points).flatten()

# Compute pointwise residual
residual_vals = laplace_interp_vals + nonlinear_term - rhs_vals
residual_grid = np.abs(residual_vals).reshape((n_grid, n_grid))

# Plot the residual heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(residual_grid, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', aspect='auto')
ax.set_title('Heatmap of PDE Residual: |Δu + u³ - rhs|')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')  # Force square plot area

# Overlay interpolation and boundary points
if len(points) > 0:
    ax.scatter(points[:, 0], points[:, 1], color='red', s=10, label='Interpolation Points')
if len(boundary_points) > 0:
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color='blue', s=10, label='Boundary Points')
ax.legend()

# Colorbar below
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.4)
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('|PDE Residual|')

plt.tight_layout()
plt.show()

