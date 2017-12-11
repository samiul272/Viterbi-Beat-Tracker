import numpy as np
def sunity(inn):

    eps= 2.220446049250313e-16 ## value taken from MATLAB
    outt=inn/np.sum(eps+inn,axis=0)

    return outt

