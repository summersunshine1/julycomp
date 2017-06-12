import os
import numpy as np

def listfiles(datadir):
    list_dirs = os.walk(datadir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for f in files:
            filepath_list.append(os.path.join(root,f))
    return filepath_list
    
def zeroNormalize(arr):
    mu = np.average(arr)
    sigma = np.std(arr)
    if sigma == 0:
        return arr
    return (arr-mu)/sigma