import normalise
import numpy as np
from normalise import normalise


def viterbi_path(prior, transmat, obslik):
    ###obslik = obslik.T
    scaled = 1
    T = obslik.shape[1]
    Q = prior.size
    delta = np.zeros((Q, T))
    psi = np.zeros((Q, T), dtype=int)
    path = np.zeros((1, T), dtype=int)
    ## = np.zeros((1, T))
    scale = np.ones((1, T))
    t = 0
    delta[:, t] = prior * obslik[:, t]
    if scaled:
        delta[:, t], n = normalise(delta[:, t])
        scale[0, t] = 1 / n

    psi[:, t] = 0
    for t in range(1, T):
        for j in range(0, Q):
            temp = delta[:, t - 1] * transmat[:, j]
            psi[j, t] = temp.argmax(axis=0)
            delta[j, t] = temp[psi[j, t]] * obslik[j, t]

        if scaled:
            delta[:, t], n = normalise(delta[:, t])
            scale[0, t] = 1 / n

    ###p[0, T - 1] = np.max(delta[:, T - 1])
    # path[0, T - 1] = np.argmax(delta[:, T - 1]) + 1
    path[0, T - 1] = np.argmax(delta[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[0, t] = psi[(path[0, t + 1]), t + 1]
    return path
