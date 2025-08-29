"""
Quickstart: Optimal Recovery – Error vs Boundary Ratio
------------------------------------------------------
This script evaluates the effect of varying the fraction of boundary points 
(in a fixed total number of collocation points) on the error of the optimal 
recovery method for a nonlinear PDE.

For each boundary ratio in `ratios`:
  - Computes max solution error (‖u - u_hat‖∞) or
  - PDE residual error (‖Δu_hat + τ(u_hat) - RHS‖∞) or
  - Boundary residual error (‖u_hat - f‖∞ on boundary)
  depending on the `res` setting.

Results:
  - Printed as LaTeX/TikZ-ready coordinate pairs: (Boundary %, Error)
  - Plotted with log-scale y-axis showing error vs boundary ratio

Configurable parameters:
  - total_points (total number of collocation points, default: 500)
  - gamma (kernel width, default: 10)
  - solution_function ("kernel", "sin", "c2")
  - res ('res' for PDE residual, 'b_res' for boundary residual, else max-norm error)
  - ratios (list of boundary fractions to test, default: [0.0, 0.1, ..., 1.0])
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from assets.Gaussian_Kernel_JIT import Kernel as KernelClass
from assets.collocation_functions import optimal_recovery

# --- Load Data ---
data = np.load("assets/quasi_centers_data.npz")
interior_points = data["interior_points"]
interior_dists = data["interior_fill_distances"]
boundary_points = data["boundary_points"]
boundary_dists = data["boundary_fill_distances"]

# --- Settings ---
res = 'b_res'  # 'res' for residual, 'b_res' for boundary residual, anythin else for \infty-norm error

total_points = 500
gamma = 10
Kernel = KernelClass(gamma)
ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # ratios to be compared
solution_function ='kernel' #choose between kernel, sin and c2
errors = []

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

# --- Functions ---
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

f_k = lambda x: Kernel.kernel_matrix(x, np.array([[0.2, 0.5]]))
laplace_f_k = lambda x: Kernel.laplace_on_x_matrix(x, np.array([[0.2, 0.5]]))

# --- solution choice logic ---
function_dict = {
    "kernel": (f_k, laplace_f_k),
    "sin": (f_sin, laplace_f_sin),
    "c2": (f_c2, laplace_f_c2)
}

f, laplace_f = function_dict[solution_function]

rhs = lambda x: laplace_f(x) + tau(f(x))
boundary_condition = f

# --- Grid for evaluation ---
test_grid_x, test_grid_y = np.meshgrid(
    np.linspace(0, 1, 50),
    np.linspace(0, 1, 50)
)
test_points = np.stack([test_grid_x.ravel(), test_grid_y.ravel()], axis=-1)
true_vals = f(test_points).flatten()

def generate_boundary_test_points(n_per_side=100):
    x = np.linspace(0, 1, n_per_side)
    y = np.linspace(0, 1, n_per_side)

    # Four edges
    bottom = np.stack([x, np.zeros_like(x)], axis=-1)
    top = np.stack([x, np.ones_like(x)], axis=-1)
    left = np.stack([np.zeros_like(y), y], axis=-1)
    right = np.stack([np.ones_like(y), y], axis=-1)

    # Combine and return
    boundary_points = np.concatenate([bottom, top, left, right], axis=0)
    return boundary_points

boundary_test_points = generate_boundary_test_points(n_per_side=100)
# --- Main Loop ---
for ratio in ratios:
    bnd_count = int(total_points * ratio)
    int_count = total_points - bnd_count

    pts = interior_points.T[:int_count]
    bpts = boundary_points.T[:bnd_count]
    z0 = np.zeros_like(f(pts).flatten())

    try:
        interpolant, coeff, _, lap_interpolant = optimal_recovery(
            pts,
            tau,
            tau_der,
            rhs,
            boundary_condition,
            z0,
            Kernel,
            boundary_points=bpts,
            use_gradient=False,
            method='gauss-newton',
            eps=1e-10
        )
    except Exception as e:
        print(f" Recovery failed at ratio {ratio:.1f}: {e}")
        errors.append(np.nan)
        continue

    if res == 'res':
        residual = np.abs(lap_interpolant(test_points) + tau(interpolant(test_points)) - rhs(test_points).flatten())
        error = np.max(residual)
    elif res == 'b_res':
        b_residual = np.abs(interpolant(boundary_test_points) - f(boundary_test_points).flatten())
        error = np.max(b_residual)
    else:
        pred_vals = interpolant(test_points)
        error = np.max(np.abs(pred_vals - true_vals))

    errors.append(error)
    print(f"✅ Interior: {int_count}, Boundary: {bnd_count}, Error: {error:.2e}")



# Compute color range for plot
# --- Print LaTeX TikZ Plot ---

for r, err in zip(ratios, errors):
    if not np.isnan(err):
        print(f"({int(r * 100)}, {err:.3e})")








# --- Plot (log-scale y-axis) ---
plt.figure(figsize=(8, 4))
plt.plot([int(r * 100) for r in ratios], errors, marker='o')
plt.xlabel('Boundary % (of 1000 total points)')
plt.ylabel('Max ' + ('Residual' if res else 'Interpolation') + ' Error')
plt.yscale('log')  # <-- Logarithmic scale on y-axis
plt.title(f'Error vs Boundary Ratio (res={res}, total={total_points})')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

