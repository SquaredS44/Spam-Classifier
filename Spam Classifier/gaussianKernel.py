import math
import numpy as np


def gaussianKernel(x1, x2, sigma):
    # Ensure that x1 and x2 are column vector
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)
    norm = sum(np.power(x1 - x2, 2))
    sim = math.exp(-norm / (2 * (np.power(sigma, 2))))
    return sim
