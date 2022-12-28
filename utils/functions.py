import numpy as np

def cos_dist(f1, f2):
    return np.dot(f1, np.transpose(f2))

def most_common(lst):
    return max(set(lst), key=lst.count)