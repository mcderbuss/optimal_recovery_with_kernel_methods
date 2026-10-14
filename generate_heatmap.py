"""
Quickstart: f-Greedy Optimal Recovery
-------------------------------------
This script applies an f-Greedy variant of the optimal recovery method
to solve -Δu + u³ = F with Gaussian kernels. Internally it uses the
equivalent equation Δu - u³ = -F.

For each iteration, it records:
  - Max solution error (‖u - u_hat‖∞)
  - PDE residual error (‖Δu_hat + τ(u_hat) - RHS‖∞)
  - Boundary residual error (‖u_hat - f‖∞ on boundary)

Results:
  Saved to: results/heatmap_<timestamp>/
  + infinity_norm_error.csv (columns: x, error)
  + pde_residual.csv (columns: x, residual)
  + error_heatmap.png (when save_plots is True)
  + residual_heatmap.png (when save_plots is True)

Configurable parameters:
  - gamma (kernel width, default: 5)
  - iterations (number of f-Greedy steps, default: 500)
  - solution_function ("sin", "c2", "kernel")
  - save_plots (save PNG files in the timestamped results directory)
  - show_plots (display plots interactively)
"""


import numpy as np
import time
import matplotlib
from numba import njit
from assets.Gaussian_Kernel_JIT import Kernel as KernelClass
from assets.collocation_functions import greedy_loop
import pandas as pd
from datetime import datetime
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap

# --- Settings ---
gamma = 5
iterations = 1000
solution_function = 'sin'  # choose between kernel, sin and c2
heatmap_cmap = 'viridis'
point_cmap = LinearSegmentedColormap.from_list('point_order', ['#ff1493', '#ffffff'])
save_plots = True
show_plots = True

matplotlib.use('TkAgg' if show_plots else 'Agg')
import matplotlib.pyplot as plt

Kernel = KernelClass(gamma)

# --- Nonlinearity ---
@njit
def tau(x):
    return - x ** 3
@njit
def tau_der(x):
    return - 3 * x ** 2
@njit
def tau_2der(x):
    return - 6 * x


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
kernel_for_sol=KernelClass(gamma)
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
#compute interpolant
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
output_directory = Path("results") / f"heatmap_{timestamp}"
output_directory.mkdir(parents=True, exist_ok=True)

# Keep both CSVs aligned to iterations that completed successfully.
completed_iterations = len(errors)
if failed_early:
    completed_iterations = int(np.count_nonzero(times))

x_values = np.arange(completed_iterations)
error_data = pd.DataFrame({
    'x': x_values,
    'error': errors[:completed_iterations]
})
residual_data = pd.DataFrame({
    'x': x_values,
    'residual': residuals[:completed_iterations]
})

error_csv_path = output_directory / "infinity_norm_error.csv"
residual_csv_path = output_directory / "pde_residual.csv"
error_data.to_csv(error_csv_path, index=False)
residual_data.to_csv(residual_csv_path, index=False)

print(f"Saved infinity-norm errors to: {error_csv_path}")
print(f"Saved PDE residuals to: {residual_csv_path}")

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

# Map both point types to their global greedy iteration (boundary points occur every fourth step).
total_points = len(points) + len(boundary_points)
point_iterations = np.arange(1, total_points + 1)
interior_iterations = point_iterations[point_iterations % 4 != 0][:len(points)]
boundary_iterations = point_iterations[point_iterations % 4 == 0][:len(boundary_points)]

# Plot the error heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_grid, extent=[0, 1, 0, 1], origin='lower', cmap=heatmap_cmap, aspect='auto')
ax.set_title('Heatmap of Error |True Solution - Interpolant|')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')  # Force square plot area

# Overlay interpolation and boundary points
if len(points) > 0:
    ax.scatter(points[:, 0], points[:, 1], marker='o', c=interior_iterations, cmap=point_cmap,
               vmin=1, vmax=total_points, edgecolors='black', linewidths=0.9, s=24,
               label='Interpolation Points')
if len(boundary_points) > 0:
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], marker='s', c=boundary_iterations,
               cmap=point_cmap, vmin=1, vmax=total_points, edgecolors='black', linewidths=0.9,
               s=44, label='Boundary Points')
ax.legend()

# Colorbar below
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.4)
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.formatter = ScalarFormatter(useMathText=True)
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
cb.set_label('Absolute Error')

plt.tight_layout()
if save_plots:
    error_plot_path = output_directory / "error_heatmap.png"
    fig.savefig(error_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved error heatmap to: {error_plot_path}")
if show_plots:
    plt.show()
else:
    plt.close(fig)


# --- Heatmap of PDE residual ---

# Reuse grid
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# Evaluate each term
interp_vals = interpolant(grid_points).flatten()
laplace_interp_vals = laplace_interpolant(grid_points).flatten()
nonlinear_term = tau(interp_vals)
rhs_vals = rhs(grid_points).flatten()

# Compute pointwise residual
residual_vals = laplace_interp_vals + nonlinear_term - rhs_vals
residual_grid = np.abs(residual_vals).reshape((n_grid, n_grid))

# Plot the residual heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(residual_grid, extent=[0, 1, 0, 1], origin='lower', cmap=heatmap_cmap, aspect='auto')
ax.set_title('Heatmap of PDE Residual: |Δu + τ(u) - rhs|')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')  # Force square plot area

# Overlay interpolation and boundary points
if len(points) > 0:
    ax.scatter(points[:, 0], points[:, 1], marker='o', c=interior_iterations, cmap=point_cmap,
               vmin=1, vmax=total_points, edgecolors='black', linewidths=0.9, s=24,
               label='Interpolation Points')
if len(boundary_points) > 0:
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], marker='s', c=boundary_iterations,
               cmap=point_cmap, vmin=1, vmax=total_points, edgecolors='black', linewidths=0.9,
               s=44, label='Boundary Points')
ax.legend()

# Colorbar below
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.4)
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.formatter = ScalarFormatter(useMathText=True)
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
cb.set_label('|PDE Residual|')

plt.tight_layout()
if save_plots:
    residual_plot_path = output_directory / "residual_heatmap.png"
    fig.savefig(residual_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved residual heatmap to: {residual_plot_path}")
if show_plots:
    plt.show()
else:
    plt.close(fig)

