import numpy as np


def emailFeatures(w):
    # total number of words in the dictionary
    n = 1899
    x = np.zeros((n,1))
    for idx in w:
        x[idx] = 1
    return x
