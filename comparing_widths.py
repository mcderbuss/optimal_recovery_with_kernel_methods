"""
Quickstart: Comparing Kernel Widths for Optimal Recovery Problem
----------------------------------------
Evaluates a kernel-based collocation method for a nonlinear PDE across different 
Gaussian kernel widths (gamma). 

Computes:
  - Max solution error
  - PDE residual error
  - Boundary residual error
  - Condition number

Results:
  Saved to CSV: results/gammas__detail_<timestamp>_<solution>.csv
  + Log-scale error plots

Configurable parameters:
  - total_points (default: 300)
  - ratio (boundary vs interior points, default: 0.25)
  - solution_function ("sin", "c2", "kernel")
  - gammas (range of kernel widths to test)
"""
import numpy as np
from numba import njit
from assets.Gaussian_Kernel_JIT import Kernel as KernelClass
from assets.collocation_functions import optimal_recovery
from assets.collocation_functions import assemble_collocation_matrix

# --- Load Pre-Generated Centres ---
data = np.load("assets/quasi_centers_data.npz")
interior_points = data["interior_points"]
interior_dists = data["interior_fill_distances"]
boundary_points = data["boundary_points"]
boundary_dists = data["boundary_fill_distances"]

# --- Settings ---
gammas = np.logspace(np.log10(0.001), np.log10(50), 100) #widths to be compared
total_points = 300
ratio = 0.25
solution_function ='c2' #choose between kernel, sin and c2

# --- Initialization
errors_default = []
errors_res = []
errors_b_res = []

# --- Nonlinearity with derivatives---
@njit
def tau(x):
    return x ** 3

@njit
def tau_der(x):
    return 3 * x ** 2

@njit
def tau_2der(x):
    return 6 * x


# --- possible solution functions with derivatives ---
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

Kernel_for_ip = KernelClass(10)
f_k = lambda x: Kernel_for_ip.kernel_matrix(x, np.array([[0.2, 0.5]]))
laplace_f_k = lambda x: Kernel_for_ip.laplace_on_x_matrix(x, np.array([[0.2, 0.5]]))

# --- solution choice logic ---
function_dict = {
    "kernel": (f_k, laplace_f_k),
    "sin": (f_sin, laplace_f_sin),
    "c2": (f_c2, laplace_f_c2)
}

f, laplace_f = function_dict[solution_function]

# --- define right hand side
rhs = lambda x: laplace_f(x) + tau(f(x))
boundary_condition = f

# --- Evaluation Grid ---
test_grid_x, test_grid_y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
test_points = np.stack([test_grid_x.ravel(), test_grid_y.ravel()], axis=-1)
true_vals = f(test_points).flatten()

# --- Boundary Test Points ---
def generate_boundary_test_points(n_per_side=100):
    x = np.linspace(0, 1, n_per_side)
    y = np.linspace(0, 1, n_per_side)
    bottom = np.stack([x, np.zeros_like(x)], axis=-1)
    top = np.stack([x, np.ones_like(x)], axis=-1)
    left = np.stack([np.zeros_like(y), y], axis=-1)
    right = np.stack([np.ones_like(y), y], axis=-1)
    return np.concatenate([bottom, top, left, right], axis=0)

boundary_test_points = generate_boundary_test_points()

# --- select collocation points from the pre-generated points at given ratio ---
results = []
bnd_count = int(total_points * ratio)
int_count = total_points - bnd_count
pts = interior_points.T[:int_count]
bpts = boundary_points.T[:bnd_count]

for gamma in gammas:
    Kernel = KernelClass(gamma)
    
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
        print(f"  Recovery failed at ratio {ratio:.1f}: {e}")
        errors_default.append(np.nan)
        errors_res.append(np.nan)
        errors_b_res.append(np.nan)
        continue

    # Default error
    pred_vals = interpolant(test_points)
    error_default = np.max(np.abs(pred_vals - true_vals))

    # Residual error
    residual = np.abs(lap_interpolant(test_points) + tau(interpolant(test_points)) - rhs(test_points).flatten())
    error_res = np.max(residual)

    # Boundary residual
    b_residual = np.abs(interpolant(boundary_test_points) - f(boundary_test_points).flatten())
    error_b_res = np.max(b_residual)

    # Store and print all errors
    errors_default.append(error_default)
    errors_res.append(error_res)
    errors_b_res.append(error_b_res)
    cond_num=np.linalg.cond(assemble_collocation_matrix(pts,Kernel,boundary_points=bpts))

    print(f"  Gamma {gamma:.1f} | Interior: {int_count:4d}, Boundary: {bnd_count:4d} | "
            f" Error: {error_default:.2e}, Res: {error_res:.2e}, B_Res: {error_b_res:.2e} "
            f"Condition Number: {cond_num}")
    
# --- Export per-gamma error results to CSV ---
import csv
from datetime import datetime

# Create timestamp string (e.g., 2025-07-26_15-30-00)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Filename with timestamp and folder
csv_filename = f"results/gammas__detail_{timestamp}_{solution_function}.csv"

# Ensure all lists are the same length as gammas
assert len(errors_default) == len(errors_res) == len(errors_b_res) == len(gammas)

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Header
    writer.writerow([
        "Gamma",
        "Max Error (Default)",
        "Residual Error",
        "Boundary Residual Error"
    ])
    # Rows
    for gamma, e_d, e_r, e_br in zip(gammas, errors_default, errors_res, errors_b_res):
        writer.writerow([gamma, e_d, e_r, e_br])

print(f"\n Results exported to: {csv_filename}")

import matplotlib.pyplot as plt

# Set figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Plot 1: Default Error
axes[0].plot(gammas, errors_default, marker='o', color='blue')
axes[0].set_title("Max Error (Default)")
axes[0].set_xlabel("Gamma")
axes[0].set_ylabel("Error")
axes[0].set_yscale('log')
axes[0].grid(True, which='both', linestyle='--')

# Plot 2: Residual Error
axes[1].plot(gammas, errors_res, marker='s', color='green')
axes[1].set_title("Residual Error")
axes[1].set_xlabel("Gamma")
axes[1].set_ylabel("Error")
axes[1].set_yscale('log')
axes[1].grid(True, which='both', linestyle='--')

# Plot 3: Boundary Residual Error
axes[2].plot(gammas, errors_b_res, marker='^', color='red')
axes[2].set_title("Boundary Residual Error")
axes[2].set_xlabel("Gamma")
axes[2].set_ylabel("Error")
axes[2].set_yscale('log')
axes[2].grid(True, which='both', linestyle='--')

# Overall title and display
plt.suptitle("Log-Scale Error vs Gamma for Collocation Method", fontsize=16)
plt.show()


