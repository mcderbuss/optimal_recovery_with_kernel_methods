"""
Quickstart: Quasi-Uniform Collocation Center Generation
-------------------------------------------------------
This script generates well-distributed (quasi-uniform) collocation points
for both the interior and boundary of the unit square [0,1]^2.

Steps:
  1. Generate large sets of candidate points:
       - Interior: uniform random points in (0,1)^2
       - Boundary: uniform random points along all four edges
  2. Select a specified number of points (n_points) using a fill-distance
     greedy approach (maximizing minimal distance to existing points)
  3. Compute and store the corresponding fill distances for each selected point
  4. Save results to a `.npz` file for later use in PDE collocation methods

Outputs:
  - interior_points, interior_fill_distances
  - boundary_points, boundary_fill_distances
  Saved in: quasi_centers_data.npz

Configurable parameters:
  - n_points: number of centers to select (default: 3000)
  - n_candidates: total candidate points to sample (default: 100000)
"""



import numpy as np
import matplotlib.pyplot as plt

def makeQuasiCenters(nMax, input):
    def computeFillDistance(SOIS, newInput, currentMinList=np.array([])):
        # Compute the squared distance matrix
        C = -2 * SOIS.T @ newInput + np.sum(SOIS * SOIS, 0)[:, np.newaxis] + np.sum(newInput * newInput, 0)
        
        if currentMinList.size == 0:
            minList = np.min(C, axis=1)
        else:
            C = np.c_[C, currentMinList]
            minList = np.min(C, axis=1)
        
        idx = np.argmax(minList)
        value = minList[idx]
        return value, idx, minList

    SOIS = input
    idx = np.argmax(np.sum(SOIS * SOIS, axis=0))
    initStates = SOIS[:, idx][:, np.newaxis]

    value, idx, minList = computeFillDistance(SOIS, initStates)
    initStates = np.c_[initStates, SOIS[:, idx][:, np.newaxis]]

    fillDistances = [value]  # Store fill distances

    for i in range(nMax):
        value, idx, minList = computeFillDistance(SOIS, initStates, minList)
        fillDistances.append(value)
        initStates = np.c_[initStates, SOIS[:, idx][:, np.newaxis]]
        print(f'fill dist: {value}')
        print(f'{(i+1)/nMax:.2%} complete')

    return initStates, np.array(fillDistances)
        
# Set parameters
n_points =  3000 # number of points for both interior and boundary

# 1. Generate interior candidates uniformly in (0,1)^2
np.random.seed(42)
n_candidates = 100000
interior_candidates = np.random.rand(2, n_candidates)

# 2. Generate boundary candidates uniformly on all 4 edges
n_boundary_per_edge = n_candidates // 4
rand_vals = np.random.rand(n_boundary_per_edge)
zeros = np.zeros_like(rand_vals)
ones = np.ones_like(rand_vals)

bottom = np.vstack((rand_vals, zeros))   # y = 0
top    = np.vstack((rand_vals, ones))    # y = 1
left   = np.vstack((zeros, rand_vals))   # x = 0
right  = np.vstack((ones, rand_vals))    # x = 1

boundary_candidates = np.hstack((bottom, top, left, right))

# 3. Select well-distributed interior and boundary points
interior_points, interior_dists = makeQuasiCenters(n_points, interior_candidates)
boundary_points, boundary_dists = makeQuasiCenters(n_points, boundary_candidates)

# Save all results to an .npz file
np.savez("quasi_centers_data.npz",
     interior_points=interior_points,
     interior_fill_distances=interior_dists,
     boundary_points=boundary_points,
     boundary_fill_distances=boundary_dists)

