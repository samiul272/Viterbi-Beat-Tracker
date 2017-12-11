import numpy as np

from numba import jit


@jit(nopython=True, cache=True)
def normalise(A):
    z = np.sum(A)
    # Set any zeros to one before dividing
    # This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0

    s = z + int(z == 0)

    M = A / s

    return (M, z)
