import numpy as np
from numba import njit, prange

# ======== JIT-compiled RBF kernel functions ========

@njit(fastmath=True)
def phi(r, width):
    return np.exp(-width * r * r)

@njit(fastmath=True)
def phi_1(r, width):
    return -2 * width * r * np.exp(-width * r * r)

@njit(fastmath=True)
def phi_2(r, width):
    return 2 * width * (2 * width * r * r - 1) * np.exp(-width * r * r)

@njit(fastmath=True)
def phi_3(r, width):
    return -4 * (width ** 2) * r * (2 * width * r * r - 3) * np.exp(-width * r * r)

@njit(fastmath=True)
def phi_4(r, width):
    gamma_sq = width ** 2
    r_sq = r * r
    return (16 * (gamma_sq * r_sq) ** 2 - 48 * gamma_sq * width * r_sq + 12 * gamma_sq) * np.exp(-width * r_sq)

@njit(fastmath=True)
def kernel_pointwise(x, y, width):
    n = x.shape[0]
    result = np.empty(n)
    for i in range(n):
        dx = x[i, 0] - y[i, 0]
        dy = x[i, 1] - y[i, 1]
        r = np.sqrt(dx * dx + dy * dy)
        result[i] = phi(r, width)
    return result

@njit(fastmath=True)
def laplace_on_x(x, y, width):
    n = x.shape[0]
    result = np.empty(n)
    for i in range(n):
        dx = x[i, 0] - y[i, 0]
        dy = x[i, 1] - y[i, 1]
        r = np.sqrt(dx * dx + dy * dy)
        r_sq = r * r
        phi_val = phi(r, width)
        phi_2_val = 2 * width * (2 * width * r_sq - 1) * phi_val
        result[i] = phi_2_val - 2 * width * phi_val
    return result
    
# ======== JIT-compiled matrix routines ========

@njit(parallel=True, fastmath=True)
def kernel_matrix(x, y, width):
    nx, ny = x.shape[0], y.shape[0]
    result = np.empty((nx, ny))
    for i in prange(nx):
        for j in range(ny):
            dx = x[i, 0] - y[j, 0]
            dy = x[i, 1] - y[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            result[i, j] = phi(r, width)
    return result

@njit(parallel=True, fastmath=True)
def laplace_on_x_matrix(x, y, width):
    nx, ny = x.shape[0], y.shape[0]
    result = np.empty((nx, ny))
    for i in prange(nx):
        for j in range(ny):
            dx = x[i, 0] - y[j, 0]
            dy = x[i, 1] - y[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            result[i, j] = phi_2(r, width) - 2 * width * phi(r, width)
    return result

@njit(parallel=True, fastmath=True)
def laplace_on_xy_matrix(x, y, width):
    nx, ny = x.shape[0], y.shape[0]
    result = np.empty((nx, ny))
    gamma_sq = width ** 2
    for i in prange(nx):
        for j in range(ny):
            dx = x[i, 0] - y[j, 0]
            dy = x[i, 1] - y[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            r_sq = r * r
            phi_val = phi(r, width)
            term1 = phi_4(r, width)
            term2 = -8 * gamma_sq * (2 * width * r_sq - 3) * phi_val
            term3 = -4 * gamma_sq * phi_val
            result[i, j] = term1 + term2 + term3
    return result

@njit(fastmath=True)
def laplace_on_xy(x, y, width):
    n = x.shape[0]
    result = np.empty(n)
    gamma_sq = width ** 2
    for i in range(n):
        dx = x[i, 0] - y[i, 0]
        dy = x[i, 1] - y[i, 1]
        r_sq = dx * dx + dy * dy
        r = np.sqrt(r_sq)
        phi_val = phi(r, width)
        term1 = (16 * (gamma_sq * r_sq)**2 - 48 * gamma_sq * width * r_sq + 12 * gamma_sq) * phi_val
        term2 = -8 * gamma_sq * (2 * width * r_sq - 3) * phi_val
        term3 = -4 * gamma_sq * phi_val
        result[i] = term1 + term2 + term3
    return result

def laplace_on_y_matrix(x, y, width):
    return laplace_on_x_matrix(y, x, width)  # x and y swapped

class Kernel:
    def __init__(self, width):
        self.width = width

    def kernel_matrix(self, x, y):
        return kernel_matrix(x, y, self.width)

    def laplace_on_x_matrix(self, x, y):
        return laplace_on_x_matrix(x, y, self.width)

    def laplace_on_y_matrix(self, x, y):
        return laplace_on_x_matrix(y, x, self.width)

    def laplace_on_xy_matrix(self, x, y):
        return laplace_on_xy_matrix(x, y, self.width)

    def kernel(self, x, y):
        return kernel_pointwise(x, y, self.width)

    def laplace_on_x(self, x, y):
        return laplace_on_x(x, y, self.width)

    def laplace_on_y(self, x, y):
        return laplace_on_x(y, x, self.width)
        
    def laplace_on_xy(self, x, y):
        return laplace_on_xy(x, y, self.width)
