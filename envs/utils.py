
import numpy as np

def manhattan_dist(a,b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])