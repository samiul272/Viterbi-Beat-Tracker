import numpy as np
def princarg(phasein):
    #phase=princarg(phasein) maps phasein into the [-pi:pi] range
    pi=3.141592653589793 ## taken from MATLAB
    return ((phasein+pi)%(-2*pi))+np.math.pi
