"""
Quickstart: Optimal Recovery with Increasing Centers
----------------------------------------------------
This script benchmarks the *optimal recovery* collocation method for a nonlinear PDE
as the number of collocation centers increases (from 0 up to `centres`).

For each total_points value, it applies optimal recovery and computes:
  - Max solution error (‖u - u_hat‖∞)
  - PDE residual error (‖Δu_hat + τ(u_hat) - RHS‖∞)
  - Boundary residual error (‖u_hat - f‖∞ on boundary)
  - Runtime per solve

Results:
  Saved to CSV: results/evolution_predetermined_points_<timestamp>_<solution>.csv
  + Plot of errors vs total points (log scale)

Configurable parameters:
  - gamma (kernel width, default: 5)
  - fixed_ratio (boundary vs interior points, default: 0.25)
  - solution_function ("sin", "c2", "kernel")
  - centres (max number of collocation points to sweep, default: 1000)
"""



import numpy as np
from numba import njit
import time
from assets.Gaussian_Kernel_JIT import Kernel as KernelClass
from assets.collocation_functions import optimal_recovery

# --- Load Data ---
data = np.load("assets/quasi_centers_data.npz")
interior_points = data["interior_points"]
boundary_points = data["boundary_points"]

# --- Settings ---
gamma = 5
fixed_ratio = 0.25
Kernel = KernelClass(gamma)
solution_function ='c2' #choose between kernel, sin and c2
centres = 1000

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

kernel_for_ip=KernelClass(10)
f_k = lambda x: kernel_for_ip.kernel_matrix(x, np.array([[0.2, 0.5]]))
laplace_f_k = lambda x: kernel_for_ip.laplace_on_x_matrix(x, np.array([[0.2, 0.5]]))

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

# --- Sweep total_points from 0 to centres ---
results = []

for total_points in range(centres+1):
    print(f"\n--- Testing total_points = {total_points} ---")

    bnd_count = int(total_points * fixed_ratio)
    int_count = total_points - bnd_count

    pts = interior_points.T[:int_count] if int_count > 0 else np.empty((0, 2))
    bpts = boundary_points.T[:bnd_count] if bnd_count > 0 else np.empty((0, 2))
    z0 = np.zeros(int_count)
    start_time = time.time()
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
            method='gauss-newton',
            eps=1e-10
        )
    except Exception as e:
        print(f"  Recovery failed: {e}")
        results.append((total_points, np.nan, np.nan, np.nan))
        continue
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    pred_vals = interpolant(test_points)
    error_default = np.max(np.abs(pred_vals - true_vals))

    residual = np.abs(lap_interpolant(test_points) + tau(pred_vals) - rhs(test_points).flatten())
    error_res = np.max(residual)

    b_residual = np.abs(interpolant(boundary_test_points) - f(boundary_test_points).flatten())
    error_b_res = np.max(b_residual)

    print(f"  Error: {error_default:.2e}, Res: {error_res:.2e}, B_Res: {error_b_res:.2e}")
    results.append((total_points, error_default, error_res, error_b_res))

# --- Export results to CSV ---
import csv
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"results/evolution_predetermined_points_{timestamp}_{solution_function}.csv"

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Total Points", "Max Abs Error", "Max Residual", "Max Boundary Residual"])
    for row in results:
        writer.writerow(row)

print(f"\nResults exported to: {csv_filename}")

import matplotlib.pyplot as plt

# Unpack results
points, errors, residuals, b_residuals = zip(*[r for r in results if not np.isnan(r[1])])

plt.figure(figsize=(10, 6))
plt.plot(points, errors, label='Max |u - u_hat|', marker='o')
plt.plot(points, residuals, label='Max PDE Residual', marker='s')
plt.plot(points, b_residuals, label='Max Boundary Residual', marker='^')

plt.xlabel("Total Number of Centers", fontsize=12)
plt.ylabel("Error (log scale)", fontsize=12)
plt.yscale("log")
plt.title("Error vs Total Points (Fixed Ratio = 0.25)", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
