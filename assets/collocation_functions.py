import numpy as np
import scipy 
import time
import inspect
import traceback
from numba import njit
from scipy.spatial import distance
import assets.helper_functions as helper_functions
from assets.helper_functions import parallel_basin_hopping
from assets.Gaussian_Kernel_JIT import Kernel
from typing import Union
import matplotlib.pyplot as plt


def assemble_collocation_matrix(points, Kernel, boundary_points=np.array([])):
    n_points = points.shape[0]
    n_boundary = boundary_points.shape[0]
    N = 2 * n_points + n_boundary
    K = np.zeros((N, N))

    K[0:2*n_points:2, 0:2*n_points:2] = Kernel.kernel_matrix(points, points)
    K[0:2*n_points:2, 1:2*n_points:2] = Kernel.laplace_on_y_matrix(points, points)
    K[1:2*n_points:2, 0:2*n_points:2] = Kernel.laplace_on_x_matrix(points, points)
    K[1:2*n_points:2, 1:2*n_points:2] = Kernel.laplace_on_xy_matrix(points, points)

    if n_boundary > 0:
        row_b = slice(2 * n_points, N)
        col_u_even = slice(0, 2 * n_points, 2)
        col_u_odd = slice(1, 2 * n_points, 2)

        K_bu = Kernel.kernel_matrix(boundary_points, points)
        K_bly = Kernel.laplace_on_y_matrix(boundary_points, points)
        K_bb = Kernel.kernel_matrix(boundary_points, boundary_points)

        K[row_b, col_u_even] = K_bu
        K[row_b, col_u_odd] = K_bly.T
        K[col_u_even, row_b] = K_bu.T
        K[col_u_odd, row_b] = K_bly
        K[row_b, row_b] = K_bb

    return K


def optimal_recovery(points, tau, tau_der, right_hand_side, boundary_condition, z0, Kernel, boundary_points = np.array([]), use_gradient = False, method = 'gauss-newton', eps = 0, tau_2der=None):
    n_points=len(points)
    if len(boundary_points) != 0:
        n_boundary = boundary_points.shape[0]
    else:
        n_boundary = 0 
    n_total = 2 * n_points + n_boundary
    A = assemble_collocation_matrix(points, Kernel, boundary_points = boundary_points) + eps * np.eye(n_total)
    L_inv = np.linalg.inv(np.linalg.cholesky(A))
    right_hand_vals = right_hand_side(points).flatten()
    
    if method == 'optimization':
        # Shared preallocated arrays
        z_inter = np.zeros(n_total, dtype=z0.dtype)
        grad_buffer = np.zeros_like(z0)
        g_vec = np.zeros(n_total, dtype=z0.dtype)
        u_vec = np.zeros(n_total, dtype=z0.dtype)
        tau_z = np.zeros(n_points, dtype=z0.dtype)
        tau_dz = np.zeros(n_points, dtype=z0.dtype)
        tau_2dz = np.zeros(n_points, dtype=z0.dtype)

        if n_boundary > 0:
            bc_vals = boundary_condition(boundary_points).ravel()
            z_inter[2 * n_points:] = bc_vals
            g_vec[2 * n_points:] = bc_vals
        
        def target_fun(z):
            tau_z[:] = tau(z)
            z_inter[0:2*n_points:2] = z
            z_inter[1:2*n_points:2] = right_hand_vals - tau_z
            y = L_inv @ z_inter
            return np.dot(y, y)

        def target_fun_gradient(z):
            tau_z[:] = tau(z)
            tau_dz[:] = tau_der(z)

            # Fill g_vec
            g_vec[0:2*n_points:2] = z
            g_vec[1:2*n_points:2] = right_hand_vals - tau_z

            # A⁻¹ g
            y = L_inv @ g_vec
            u = L_inv.T @ y

            # Gradient: 2 * (u_even - tau'(z) * u_odd)
            u_even = u[0:2*n_points:2]
            u_odd = u[1:2*n_points:2]
            grad_buffer[:] = 2 * (u_even - tau_dz * u_odd)
            return grad_buffer

        def target_fun_hessian(z):
            tau_z[:] = tau(z)
            tau_dz[:] = tau_der(z)
            tau_2dz[:] = tau_2der(z)

            g_vec[0:2*n_points:2] = z
            g_vec[1:2*n_points:2] = right_hand_vals - tau_z

            y = L_inv @ g_vec
            u = L_inv.T @ y
            u_even = u[0:2*n_points:2]
            u_odd = u[1:2*n_points:2]

            H1_diag = 2 * (u_even + tau_dz**2 * u_odd)
            H2_diag = -2 * g_vec[1:2*n_points:2] * tau_2dz

            return np.diag(H1_diag + H2_diag)
        
        # Optimize to find z minimizing target_fun
        if use_gradient:
            minimizer_kwargs = {
                "method": "trust-constr",
                "jac": target_fun_gradient,
                "hess": target_fun_hessian,
                "options": {
                    "gtol": 1e-8,
                    "xtol": 1e-10,
                    "verbose": 0
                }
            }
        else:
            minimizer_kwargs = {
                    "method": "L-BFGS-B",  # Robust local minimizer
                    "options": {
                        "ftol": 1e-9,      # Very tight function tolerance
                        "gtol": 1e-5,      # Gradient tolerance
                        "disp": False
                    }
                }
        res = parallel_basin_hopping(target_fun, z0, minimizer_kwargs, n=50, stepsize=2.0)

        # Compute optimal interlaced solution
        # Compute the optimal interlaced solution
        optimal_z = res.x
        
    elif method == 'gauss-newton':
        #pre allocate memory and precompute values
        z_inter = np.zeros(n_total, dtype=z0.dtype)
        if n_boundary > 0:
            bc_vals = boundary_condition(boundary_points).ravel()
            z_inter[2 * n_points:] = bc_vals
            
        J = np.zeros((n_total, n_points))
        J[0:2*n_points:2, :] = np.eye(n_points)
        
        def F_bar(z):
            tau_z = tau(z)
            z_inter[0:2*n_points:2] = z
            z_inter[1:2*n_points:2] = right_hand_vals - tau_z
            return z_inter
        
        def grad_F_bar(z):
            J[1:2*n_points:2, :] = -np.diag(tau_der(z))
            return J
        
        optimal_z = z0
        
        for l in range(100):
            F_l = F_bar(optimal_z)
            grad_F_l = grad_F_bar(optimal_z)
            M = L_inv @ grad_F_l         # A = L⁻¹ ∇F_l
            b = L_inv @ F_l              # b = L⁻¹ F_l
            delta_z, *_ = np.linalg.lstsq(M, -b, rcond=None)
            delta_abs = np.linalg.norm(delta_z)
            optimal_z += delta_z
            if delta_abs < 10e-7:
                print('gauss-newton converged')
                break
            
            
        
    z_dash_opt = right_hand_side(points) - tau(optimal_z).reshape(-1, 1)
    z_opt = np.empty(2 * n_points + n_boundary, dtype=z0.dtype)
    z_opt[::2][:n_points] = optimal_z
    z_opt[1::2][:n_points] = z_dash_opt.T
    if n_boundary > 0:
        z_opt[2 * n_points:] = boundary_condition(boundary_points).T
        
    coeff=scipy.linalg.solve(A, z_opt)
    
    def interpolant(x):
        even_coeffs = coeff[0:2*n_points:2]  
        odd_coeffs = coeff[1:2*n_points:2]   
        boundary_coeffs = coeff[2*n_points:2*n_points+n_boundary]
        inter_result = np.dot(Kernel.kernel_matrix(points,x).T,even_coeffs) 
        if len(boundary_points) != 0:
            boundary_result = np.dot(Kernel.kernel_matrix(boundary_points,x).T,boundary_coeffs)
        else: 
            boundary_result = 0
        lap_result = np.dot(Kernel.laplace_on_x_matrix(points,x).T,odd_coeffs)
        return inter_result + boundary_result + lap_result
    
    def laplace_interpolant(x):
        even_coeffs = coeff[0:2*n_points:2] 
        odd_coeffs = coeff[1:2*n_points:2]  
        boundary_coeffs = coeff[2*n_points:2*n_points+n_boundary]
        inter_result = np.dot(Kernel.laplace_on_x_matrix(points,x).T,even_coeffs) 
        if len(boundary_points) != 0:
            boundary_result = np.dot(Kernel.laplace_on_x_matrix(boundary_points,x).T,boundary_coeffs)
        else:
            boundary_result = 0
        lap_result = np.dot(Kernel.laplace_on_xy_matrix(points,x).T,odd_coeffs)
        return inter_result + boundary_result + lap_result
    return interpolant, coeff, z_opt, laplace_interpolant
    
def greedy_loop(
    tau,
    tau_der,
    right_hand_side,
    boundary_condition,
    iter,
    Kernel,
    points=np.array([]),
    boundary_points=np.array([]),
    sol=None,
    verbosity=False,
    eps = 0
):
    errors = np.zeros(iter)
    residuals = np.zeros(iter)
    boundary_residuals = np.zeros(iter)
    failed_early = False
    times = np.zeros(iter)

    def worst_res_interior(interpolant, laplace_interpolant, grid_size=100):
        def res(x):
            return np.abs(laplace_interpolant(x) + tau(interpolant(x)) - right_hand_side(x).flatten())
        grid_x = np.linspace(1 / grid_size, 1 - 1 / grid_size, grid_size)
        grid_y = np.linspace(1 / grid_size, 1 - 1 / grid_size, grid_size)
        grid_points = np.array(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T
        residuals = res(grid_points)
        worst_index = np.argmax(residuals)
        return grid_points[worst_index], residuals[worst_index]

    def worst_res_boundary(interpolant, grid_size=100):
        boundary_x = np.linspace(0, 1, grid_size)
        boundary_y = np.linspace(0, 1, grid_size)
        left = np.column_stack([np.zeros(grid_size), boundary_y])
        right = np.column_stack([np.ones(grid_size), boundary_y])
        bottom = np.column_stack([boundary_x, np.zeros(grid_size)])
        top = np.column_stack([boundary_x, np.ones(grid_size)])
        boundary_points = np.vstack([left, right, bottom, top])
        residuals = np.abs(interpolant(boundary_points) - boundary_condition(boundary_points).flatten())
        worst_index = np.argmax(residuals)
        return boundary_points[worst_index], residuals[worst_index]

    if len(points) == 0 and len(boundary_points) == 0:
        interpolant = lambda x: np.array([0.])
        laplace_interpolant = lambda x: np.array([0.])
        points = np.array([])
        boundary_points = np.array([])
    elif len(points) > 0:
        interpolant, coeff, z_opt, laplace_interpolant = optimal_recovery(
            points, tau, tau_der, right_hand_side, boundary_condition,
            np.zeros(points.shape[0]), Kernel, boundary_points=boundary_points, eps = eps
        )
    else:
        raise ValueError("When boundary points are given, interior points must also be provided.")

    for i in range(iter):
        start_time = time.time()
        if verbosity:
            print(f'Iteration {i + 1} out of {iter}.')

        worst_interior = worst_res_interior(interpolant, laplace_interpolant)
        worst_boundary = worst_res_boundary(interpolant)

        if  (i+1) %4 != 0:
            points = worst_interior[0].reshape(1, -1) if len(points) == 0 else np.vstack([points, worst_interior[0]])
        else:
            boundary_points = worst_boundary[0].reshape(1, -1) if len(boundary_points) == 0 else np.vstack([boundary_points, worst_boundary[0]])

        residuals[i] = abs(worst_interior[1]).item()
        boundary_residuals[i] = abs(worst_boundary[1]).item()

        if verbosity:
            print(f'Residual before iteration: {residuals[i]}')
            print(f'Boundary residual before iteration: {boundary_residuals[i]}')

        try:
            interpolant, coeff, z_opt, laplace_interpolant = optimal_recovery(
                points, tau, tau_der, right_hand_side, boundary_condition,
                interpolant(points).flatten(), Kernel, boundary_points=boundary_points, eps = eps
            )
        except RuntimeError:
            if verbosity:
                print("Warning: Optimization failed - terminating loop early")
                traceback.print_exc()
            failed_early = True
            break
        except Exception as e:
            if verbosity:
                print(f"Unexpected exception occurred: {e}")
                traceback.print_exc()
            failed_early = True
            break

        if sol is not None:
            func = lambda x: abs(interpolant(x) - sol(x).flatten())
            grid_x = np.linspace(1 / 100, 1 - 1 / 100, 100)
            grid_y = np.linspace(1 / 100, 1 - 1 / 100, 100)
            grid_points = np.array(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T
            vals = func(grid_points)
            errors[i] = np.max(vals)
            if verbosity:
                print(f'Sup-norm error after iteration: {errors[i]}')

        end_time = time.time()
        times[i] = end_time - start_time
        if verbosity:
            print(f'Iteration took {times[i]:.4f} seconds')

    return interpolant, coeff, points, boundary_points, residuals, boundary_residuals, errors, failed_early, times, laplace_interpolant
