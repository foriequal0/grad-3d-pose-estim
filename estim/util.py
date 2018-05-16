import numpy as np
from scipy import io as sio


def mrdivide(a, b):
    # return a @ np.linalg.pinv(b)
    return np.linalg.lstsq(b.T, a.T, rcond=None)[0].T


def mldivide(a, b):
    # return np.linalg.pinv(a) @ b
    return np.linalg.lstsq(a, b, rcond=None)


def centralize(S):
    if np.prod(np.shape(S)) != 0:
        return S - np.mean(S, 1, keepdims=True)
    else:
        return np.array([])


def reshapeS(S, mode):
    if mode == 'b2v':
        F, P = np.shape(S)
        F = F // 3
        return np.reshape(S.T, [3 * P, F])
    elif mode == 'v2b':
        P, F = np.shape(S)
        P = P // 3
        return np.reshape(S.T, [P, 3 * F])
    else:
        raise ValueError("Invalid mode: {}".format(mode))


def load(path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False, mat_dtype=True)


def findWmax(hm):
    #TODO: batch version
    W_max = np.zeros([2, np.size(hm, 0)])
    score = np.zeros(np.size(hm, 0))
    for i in range(np.size(hm, 0)):
        max = np.max(hm[i])
        score[i] = max
        [u,v] = np.where(hm[i] == max)
        W_max[:, i] = [v[0],u[0]]
    return W_max, score