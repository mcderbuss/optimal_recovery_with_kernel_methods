import numpy as np
import scipy
import math
import os
from datetime import datetime
from scipy.optimize import minimize_scalar, differential_evolution
from itertools import product
from concurrent.futures import ThreadPoolExecutor

def array_factorial(array):
    result = 1
    for a in array:
        result *= math.factorial(a)
    
    return result

def multinomial_coefficient(alpha, beta):
    """
    Compute the multinomial coefficient for two vectors alpha and beta.

    Parameters:
        alpha (list): A list of non-negative integers [alpha_1, alpha_2, ..., alpha_l].
        beta (list): A list of non-negative integers [beta_1, beta_2, ..., beta_l].

    Returns:
        int: The multinomial coefficient for the two vectors.
    """
    if len(alpha) != len(beta):
        raise ValueError("Vectors alpha and beta must have the same length.")

    alpha_min_beta = np.array(alpha, dtype = int) - np.array(beta, dtype = int)
    
    return array_factorial(alpha) // (array_factorial(beta) * array_factorial(alpha_min_beta))

def count_occurrences(alpha, d):
    """
    Count the occurrences of each integer between 0 and d-1 in the list alpha.

    Args:
        alpha: List of integers between 0 and d-1.
        d: The length of the output vector (maximum integer value + 1).

    Returns:
        A list of length d where the i-th entry is the number of occurrences of i in alpha.
    """
    # Initialize a vector of zeros with length d
    occurrence_vector = [0] * d

    # Count occurrences of each integer in alpha
    for i in alpha:
        if 0 <= i < d:
            occurrence_vector[i] += 1
        else:
            raise ValueError(f"Invalid value {i} in alpha. All values must be between 0 and {d-1}.")

    return occurrence_vector

def reconstruct_alpha(occurrence_vector):
    """
    Reconstruct the original list of integers alpha from the occurrence vector.

    Args:
        occurrence_vector: A list of length d where the i-th entry is the number of occurrences of i.

    Returns:
        A list of integers alpha, where each integer i appears occurrence_vector[i] times.
    """
    return [i for i, count in enumerate(occurrence_vector) for _ in range(count)]

def kron_del(i,j):
    if i == j:
        return 1
    else:
        return 0


def unit_square_grid(m):
    """
    Create a (m+1)^2 grid of equidistant points in the unit square [0,1]x[0,1].
    
    Parameters:
        m (int): Determines the number of points along each axis (m+1 points)
    
    Returns:
        np.ndarray: Array of shape ((m+1)^2, 2) containing the grid points
    """
    # Create 1D array of equidistant points along each axis
    
    points_1d = np.linspace(0, 1, m+1)
    
    # Create 2D grid
    x, y = np.meshgrid(points_1d, points_1d)
    
    # Stack and reshape to get (x,y) pairs
    grid = np.column_stack((x.ravel(), y.ravel()))
    
    return grid
    

def interior_unit_square_grid(m):
    """
    Create a (m-1)^2 grid of equidistant points in the interior of the unit square (0,1)x(0,1).
    Excludes all boundary points.
    
    Parameters:
        m (int): Determines the number of divisions (produces m+1 points along each axis,
                 but returns only (m-1)^2 interior points
    
    Returns:
        np.ndarray: Array of shape ((m-1)^2, 2) containing the interior grid points
    """
    if m < 2:
        raise ValueError("m must be at least 2 to have interior points")
    
    # Create 1D array of equidistant points along each axis
    points_1d = np.linspace(0, 1, m+1)
    
    # Get interior points (exclude first and last elements)
    interior_points = points_1d[1:-1]
    
    # Create 2D grid of interior points
    x, y = np.meshgrid(interior_points, interior_points)
    
    # Stack and reshape to get (x,y) pairs
    grid = np.column_stack((x.ravel(), y.ravel()))
    
    return grid
    

def equidistant_boundary_points(m):
    """
    Generate 4m equidistant points on the boundary of the unit square [0,1] x [0,1].

    Parameters:
        m (int): Number of points per side (total points = 4m).

    Returns:
        np.ndarray: Array of shape (4m, 2) containing the boundary points.
    """
    # Generate points for each side (excluding the last corner to avoid duplication)
    # Bottom side (y = 0, x varies from 0 to 1)
    bottom = np.column_stack((np.linspace(0, 1, m, endpoint=False), np.zeros(m)))
    
    # Right side (x = 1, y varies from 0 to 1)
    right = np.column_stack((np.ones(m), np.linspace(0, 1, m, endpoint=False)))
    
    # Top side (y = 1, x varies from 1 to 0)
    top = np.column_stack((np.linspace(1, 0, m, endpoint=False), np.ones(m)))
    
    # Left side (x = 0, y varies from 1 to 0)
    left = np.column_stack((np.zeros(m), np.linspace(1, 0, m, endpoint=False)))
    
    # Combine all sides (order: bottom → right → top → left)
    boundary_points = np.vstack((bottom, right, top, left))
    
    return boundary_points

def find_max_on_boundary(f, verbose=False, global_opt=False):
    """
    Find the maximum of f(x,y) on the boundary of [0,1]x[0,1].
    
    Parameters:
        f: function f([x,y]) to maximize
        verbose: whether to print results
        global_opt: if True, uses global optimization (for non-convex problems)
    
    Returns:
        (max_xy, max_val): tuple of (coordinates, maximum value)
    """
    edges = [
        (lambda y: np.array([0, y]), (0, 1)),    # left
        (lambda y: np.array([1, y]), (0, 1)),    # right
        (lambda x: np.array([x, 0]), (0, 1)),    # bottom
        (lambda x: np.array([x, 1]), (0, 1))     # top
    ]
    
    if global_opt:
        def optimize(edge_func, bounds):
            res = differential_evolution(lambda t: -f(edge_func(t)), [bounds])
            return {'x': res.x[0], 'fun': res.fun}
    else:
        def optimize(edge_func, bounds):
            res = minimize_scalar(lambda t: -f(edge_func(t)), bounds=bounds, method='bounded')
            return {'x': res.x, 'fun': res.fun}

    max_val = -np.inf
    max_xy = None
    for edge_func, bounds in edges:
        res = optimize(edge_func, bounds)
        current_val = -res['fun']
        if current_val > max_val:
            max_val = current_val
            max_xy = edge_func(res['x'])

    if verbose:
        print(f"Max on boundary: {max_val:.6f} at {max_xy}")
    
    return max_xy, max_val
    
def max_on_unit_square(func, m):
    """
    Find the maximum value of a function func(x) over the unit square [0,1]x[0,1],
    evaluated on a (m+1)^2 grid.

    Parameters:
        func (function): Function that takes a 2D vector [x, y].
        m (int): Number of divisions along each axis (m+1 points).

    Returns:
        max_val (float): Maximum value found.
        max_point (np.ndarray): 2D vector where maximum was found.
    """
    grid = unit_square_grid(m)

    max_val = -np.inf
    max_point = None

    for point in grid:
        val = abs(func(point.reshape(1, -1)))
        if val > max_val:
            max_val = val

    return max_val


def save_arrays(coeff, points, boundary_points, residuals, boundary_residuals, errors, note, runtime, times, gamma):
    """
    Save arrays, note, runtime, and per-iteration times to a timestamped .npz file

    Args:
        coeff: numpy array of coefficients
        points: numpy array of interior points
        boundary_points: numpy array of boundary points
        residuals: numpy array of interior residuals
        boundary_residuals: numpy array of boundary residuals
        errors: numpy array of errors
        note: string comment to store with the data
        runtime: float, total execution time in seconds
        times: numpy array of time per iteration

    Returns:
        str: path to the saved file
    """
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"results/{timestamp}.npz"

    np.savez_compressed(
        filename,
        coeff=coeff,
        points=points,
        boundary_points=boundary_points,
        residuals=residuals,
        boundary_residuals=boundary_residuals,
        errors=errors,
        note=np.array([note]),
        runtime=np.array([runtime]),
        times=times,
        gamma=gamma
    )
    return filename

def load_arrays(filename):
    """
    Load saved arrays and metadata from a .npz file

    Args:
        filename: path to the .npz file

    Returns:
        dict: Dictionary containing arrays, note string, runtime, and per-iteration times
    """
    data = np.load(filename, allow_pickle=True)
    result = {
        'coeff': data['coeff'],
        'points': data['points'],
        'boundary_points': data['boundary_points'],
        'residuals': data['residuals'],
        'boundary_residuals': data['boundary_residuals'],
        'errors': data['errors'],
        'note': str(data['note'][0]) if 'note' in data else "",
        'runtime': float(data['runtime'][0]) if 'runtime' in data else None,
        'times': data['times'] if 'times' in data else None,
        'gamma': data['gamma'] if 'gamma' in data else 5
    }
    return result
    
            

def parallel_basin_hopping(target_fun, z0, minimizer_kwargs, n=10, stepsize=2.0, seed=None, use_processes=False):
    """
    Drop-in replacement for scipy.optimize.basinhopping using parallel random restarts.

    Args:
        target_fun (callable): Objective function.
        z0 (ndarray): Initial guess.
        minimizer_kwargs (dict): Passed to scipy.optimize.minimize.
        n (int): Number of random restarts.
        stepsize (float): Perturbation scale for generating new starting points.
        seed (int or None): Random seed.
        use_processes (bool): Use processes instead of threads (slower startup, better isolation).
    
    Returns:
        OptimizeResult: The best result found among all parallel runs.
    """
    rng = np.random.default_rng(seed)
    z0 = np.asarray(z0)
    
    # Create perturbed initial guesses
    initial_points = [z0 + stepsize * rng.standard_normal(z0.shape) for _ in range(n)]
    initial_points[0] = z0  # Include original starting point

    def run_minimizer(start):
        return scipy.optimize.minimize(target_fun, start, **minimizer_kwargs)
    
    Executor = ThreadPoolExecutor if not use_processes else concurrent.futures.ProcessPoolExecutor
    with Executor() as executor:
        results = list(executor.map(run_minimizer, initial_points))
    
    # Select the result with the lowest objective value
    best_result = min(results, key=lambda res: res.fun)
    return best_result

