import numpy as np
import scipy.interpolate as interp


def interpolate1D(arr, new_length):
    '''
    :param arr: the array to interpolate
    :param new_length: the new length for the array arr,
    :return: a resized version of arr, with interpolated values inserted
    '''
    N = new_length
    L = arr.shape[0]
    if N == L:
        return arr
    assert N > L, "To downsample, use something else"

    stepsize = float(N)/float(L)
    xcoords = [stepsize * float(i) for i in np.arange(L)]
    # --> equally spaced (non-integer) coordinates in range(N)
