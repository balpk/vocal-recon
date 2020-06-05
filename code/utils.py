import numpy as np
import scipy.interpolate as interp
import os

def try_extract_directory(path):
    path_ = path.split("/")
    if "." in path_[-1]:
        return "/".join(path_[:-1])
    else:
        return path

def ensure_dir(path):
    '''https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory'''
    import os
    directory = try_extract_directory(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


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
    assert N > L, "To downsample, use something else. Or remove the signal strength array altogether, we don't use it anyway, right?"

    stepsize = float(N)/(float(L - 1))
    xcoords = [stepsize * float(i) for i in np.arange(L)]
    # --> equally spaced (non-integer) coordinates in range(N+1), ending at N and starting at 0

    interpolating_f = interp.interp1d(xcoords, arr, kind="linear", axis=0)
    new_xcoords = np.arange(N)
    new_arr = interpolating_f(new_xcoords)
    return new_arr


def test_interpolate1D():
    arr = np.array([1,3,5,7,9,10,11,10,9])
    arr2 = np.array([(i - 3)**2 for i in np.arange(9)])
    array_2d = np.hstack((arr[..., np.newaxis], arr2[..., np.newaxis]))
    N = 14 # less than twice as much; no common denominator? (Teiler)
    L = len(arr)
    import matplotlib.pyplot as plt

    #plt.figure()
    new_arr = interpolate1D(arr, N)
    new_arr_2d = interpolate1D(array_2d, N)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(list(np.arange(len(arr))), arr, color="g")
    ax[0, 0].plot(list(np.arange(N)), new_arr)
    ax[0, 1].plot(list(np.arange(len(arr))), arr, color="g")
    ax[0, 1].plot(np.array(list(np.arange(N))) * ((L-1) / N), new_arr)

    ax[1, 0].plot(list(np.arange(L)), array_2d[:,0], color="orange")
    ax[1, 0].plot(list(np.arange(L)), array_2d[:,1], color="y")
    ax[1, 0].plot(list(np.arange(N)), new_arr_2d[:,0], color="r")
    ax[1, 0].plot(list(np.arange(N)), new_arr_2d[:,1], color="g")
    ax[1, 1].plot(np.array(list(np.arange(L))) * N/(L-1), array_2d[:,0], color="orange")
    ax[1, 1].plot(np.array(list(np.arange(L))) * N/(L-1), array_2d[:,1], color="y")
    ax[1, 1].plot(list(np.arange(N)), new_arr_2d[:,0], color="r")
    ax[1, 1].plot(list(np.arange(N)), new_arr_2d[:,1], color="g")
    plt.pause(0.001)

    print("Beau-ti-ful.")


if __name__ == '__main__':
    test_interpolate1D()