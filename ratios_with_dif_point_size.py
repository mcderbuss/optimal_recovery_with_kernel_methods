"""
Quickstart: Optimal Recovery – Boundary/Interior Ratio Sweep
-----------------------------------------------------------
This script evaluates the effect of varying the ratio of boundary to interior
collocation points in the optimal recovery method for a nonlinear PDE.

For each total number of points (100 → 2000, step 100) and each boundary ratio:
  - Computes max solution error (‖u - u_hat‖∞)
  - Computes PDE residual error (‖Δu_hat + τ(u_hat) - RHS‖∞)
  - Computes boundary residual error (‖u_hat - f‖∞ on boundary)
  - Identifies the boundary ratio minimizing each error type

Results:
  Saved to CSV: results/optimal_ratio_over_points_<timestamp>_<solution>.csv
  + Console summary of best ratios vs total points

Configurable parameters:
  - gamma (kernel width, default: 10)
  - solution_function ("kernel", "sin", "c2")
  - ratios (list of boundary point fractions to test, default: [0.0, 0.1, ..., 1.0])
  - total points range and step (default: 100 → 2000, step 100)
"""

import numpy as np
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
gamma = 10
Kernel = KernelClass(gamma)
ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #percentages of boundary points to be evaluated
solution_function ='kernel' #choose between kernel, sin and c2

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

# --- Sweep total_points from 100 to 2000 ---
results = []

for total_points in range(100, 2001, 100):
    print(f"\n--- Testing total_points = {total_points} ---")
    errors_default = []
    errors_res = []
    errors_b_res = []

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

        print(f"  Ratio {ratio:.1f} | Interior: {int_count:4d}, Boundary: {bnd_count:4d} | "
              f"Error: {error_default:.2e}, Res: {error_res:.2e}, B_Res: {error_b_res:.2e}")

    errors_default_np = np.array(errors_default)
    errors_res_np = np.array(errors_res)
    errors_b_res_np = np.array(errors_b_res)
    max_res_b_res_np = np.maximum(errors_res_np, errors_b_res_np)

    def get_best(err_array):
        if np.all(np.isnan(err_array)):
            return None, np.nan
        idx = np.nanargmin(err_array)
        return ratios[idx], err_array[idx]

    
    best_ratio_default, min_err_default = get_best(errors_default_np)
    best_ratio_res, min_err_res = get_best(errors_res_np)
    best_ratio_b_res, min_err_b_res = get_best(errors_b_res_np)
    best_ratio_max_res_b_res, min_err_max_res_b_res = get_best(max_res_b_res_np)

    results.append((
    total_points,
    best_ratio_default, min_err_default,
    best_ratio_res, min_err_res,
    best_ratio_b_res, min_err_b_res,
    best_ratio_max_res_b_res, min_err_max_res_b_res))

# --- Summary ---
print("\n=== Summary of Best Ratios ===")
print(f"{'Points':>8} {'Best Ratio':>12} {'Min Error':>15} "
      f"{'Best ResRatio':>15} {'Min ResError':>15} "
      f"{'Best BResRatio':>15} {'Min BResError':>15} "
      f"{'Best MaxResBResRatio':>20} {'Min MaxResBResError':>20}")

for (tp, br_d, me_d, br_r, me_r, br_br, me_br, br_mrb, me_mrb) in results:
    print(f"{tp:8} {str(br_d):>12} {me_d:15.2e} "
          f"{str(br_r):>15} {me_r:15.2e} "
          f"{str(br_br):>15} {me_br:15.2e} "
          f"{str(br_mrb):>20} {me_mrb:20.2e}")

# --- Export results to CSV ---
import csv
from datetime import datetime

# Create timestamp string (e.g., 2025-07-26_15-30-00)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Filename with timestamp and folder
csv_filename = f"results/optimal_ratio_over_points_{timestamp}_{solution_function}.csv"

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Header
    writer.writerow([
        "Total Points",
        "Best Ratio (Default)", "Min Error (Default)",
        "Best Ratio (Residual)", "Min Error (Residual)",
        "Best Ratio (Boundary Residual)", "Min Error (Boundary Residual)",
        "Best Ratio (Max of Res & BRes)", "Min Error (Max of Res & BRes)"
    ])
    # Rows
    for (tp, br_d, me_d, br_r, me_r, br_br, me_br, br_mrb, me_mrb) in results:
        writer.writerow([tp, br_d, me_d, br_r, me_r, br_br, me_br, br_mrb, me_mrb])

print(f"\n Results exported to: {csv_filename}")

