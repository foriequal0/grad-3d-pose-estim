import scipy.io as sio
import os.path as path
import h5py
import numpy as np
from object3d.opts import opts
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions
from pymanopt import Problem
from pymanopt.tools.multi import multiprod
from pathlib import Path


def load(path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False, mat_dtype=True)

datapath = path.join(opts.data_dir, "pascal3d/annot")
predpath = 'exp/pascal3d/'
savepath = 'result/cad/'

cad_specific = False

annotfile = path.join(datapath, "valid.mat")
annotmat = load(annotfile)
annot = annotmat["annot"]


def normalizeS(S):
    t = np.mean(S, axis=1, keepdims=True)
    S = S - t * np.ones([1, np.size(S, 1)])
    a = np.mean(np.std(S, axis=1))
    S = S/a
    return S


def getPascalTemplate(model):
    p = len(model.pnames)

    S = np.full([3, p], np.nan)
    for j in range(p):
        xyz = getattr(model, model.pnames[j])
        if np.all(np.shape(xyz) != 0):
            S[:, j] = xyz[:]

    kpt_idx = np.where(np.mean(~np.isnan(S), axis=0) == 1)[0]
    kpt_name = model.pnames[kpt_idx]
    S = S[:,kpt_idx]

    B = normalizeS(S)
    return {
        "B": B,
        "mu": B,
        "pc": [],
        "kpt_id": kpt_idx + 1,
        "kpt_name": kpt_name,
        "model_id": 1
    }

def findWmax(hm):
    #TODO: batch version
    W_max = np.zeros([2, np.size(hm, 0)])
    score = np.zeros(np.size(hm, 0))
    for i in range(np.size(hm, 0)):
        max = np.max(hm[i])
        score[i] = max
        [u,v] = np.where(hm[i] == max)
        W_max[:, i] = [v,u]
    return W_max, score

def mrdivide(a, b):
    # return a @ np.linalg.pinv(b)
    return np.linalg.lstsq(b.T, a.T)[0].T

def prox_2norm(Z, lam):
    [U,W,V] = np.linalg.svd(Z, full_matrices=False)
    w = W
    if np.sum(w) <= lam:
        w = [0,0]
    elif w[0]-w[1] <= lam:
        w = [(np.sum(w) - lam)/2, (np.sum(w) - lam)/2]
    else:
        w = [w[0] - lam, w[1]]
    X = U @ np.diag(w) @ V
    normX = w[0]
    return X, normX

def proj_deformable_approx(X):
    r = np.size(X, axis=0)
    d = r // 3

    A = np.zeros([3,3])
    for i in range(d):
        Ai = X[i*3:(i+1)*3, :]
        A = A + Ai @ Ai.T

    [U,S,V] = np.linalg.svd(A)

    Q = U[:, 0:2]

    G = np.zeros([2,2])
    for i in range(d):
        Ai = X[i*3:(i+1)*3, :]
        Ti = Q.T @ Ai
        gi = np.array([[np.trace(Ti)], [Ti[1,0] - Ti[0,1]]])
        G = G + gi @ gi.T

    [U1, S1, V1] = np.linalg.svd(G)
    G = np.zeros([2,2])
    for i in range(d):
        Ai = X[i*3:(i+1)*3, :]
        Ti = Q.T @ Ai
        gi = np.array([[Ti[0,0] - Ti[1,1]], [Ti[1,0] - Ti[0,1]]])
        G = G + gi @ gi.T

    [U2, S2, V2] = np.linalg.svd(G)

    if (S1[0] > S2[1]):
        u = U1[:, 0]
        R = np.array([[u[0], -u[1]], [u[1], u[0]]])
    else:
        u = U2[:, 0]
        R = np.array([[u[0], u[1]], [u[1], -u[0]]])

    Q = Q @ R

    Y = []
    L = []
    for i in range(d):
        Ai = X[i*3:(i+1)*3, :]
        ti = 0.5 * np.trace(Q.T @ Ai)

        L.append([ti])
        Y.append([ti * Q])
    return np.array(Y), np.array(L), Q

def syncRot(T):
    [_, L, Q] = proj_deformable_approx(T.T)
    s = np.sign(L[np.where(np.abs(L) == np.max(np.abs(L), axis=0))])
    C = s * L.T
    R = s * Q.T
    R = np.append(R, [np.cross(R[0,:], R[1,:])], axis=0)
    return R, C

def estimateR_weighted(S, W, D, R0):
    #TODO: batched version
    A = S.T
    B = W.T
    X0 = R0[0:2, :].T

    [m,n] = np.shape(A)
    N=1
    p = np.size(B, 1)

    At = np.zeros([n, m])
    At[:, :] = A[:,:].T

    previous = {}
    def print_only_if_diff(key, value):
        if key not in previous:
            print(key, " = ", value)
            previous[key] = value.copy() if hasattr(value, 'copy') else value
            return
        if not np.array_equal(previous[key], value):
            print(key, " = ", value)
            previous[key] = value.copy() if hasattr(value, 'copy') else value
            return
    print("pref:")
    for (k, v) in [('A', A), ('B', B), ('D', D), ('N', N)]:
        print_only_if_diff(k, v)
    def cost(X):
        E = A @ X - B
        f = np.trace(E.T @ D @ E) / (2*N)
        print("cost: ")
        for (k, v) in [('A', A), ('B', B), ('X', X), ('E', E), ('D', D),
                       ('N', N), ('f', f)]:
            print_only_if_diff(k, v)
        return f

    def grad(X):
        E = A @ X - B
        egrad = (At /N) @ (D @ E)
        g = manifold.egrad2rgrad(X, egrad)
        print("grad: ")
        for (k, v) in [('A', A), ('B', B), ('X', X), ('E', E), ('D', D),
                       ('At', At), ('egrad', egrad), ('g', g)]:
            print_only_if_diff(k, v)
        return g

    manifold = Stiefel(n, p)
    problem = Problem(manifold, cost=cost, grad=grad)
    solver = TrustRegions(maxiter=10, mingradnorm=1e-3)
    np.set_printoptions(precision=4, suppress=True)
    X = solver.solve(problem, X0)
    return X.T

def estimateC_weighted(W, R, B, D, lam):
    P = np.size(W, 1)
    K = np.size(B, 0)//3

    d = np.diag(D)
    D = np.zeros([2 * P, 2 * P])
    for i in range(P):
        D[2*i, 2*i] = d[i]
        D[2*i+1, 2*i+1] = d[i]

    y = W.flatten('F')
    X = np.zeros([2 * P, K])
    for k in range(K):
        RBk = R @ B[3 * k:3 * (k + 1), :]
        X[:, k] = RBk.flatten('F')
    C = np.linalg.pinv(X.T @ D @ X + lam * np.eye(np.size(X, 1))) @ X.T @ D @ y
    return C.T

def PoseFromKpts_WP(W, d, weight=None, verb=True, lam = 1, tol=1e-3):
    B = d["mu"]
    pc = d["pc"]

    [k,p] = np.shape(B)
    k = k // 3

    alpha = 1
    D = np.eye(p) if weight is None else np.diag(weight)

    # centralize basis
    B = B - np.mean(B, axis=1, keepdims=True)

    # initialization
    M = np.zeros([2, 3*k])
    C = np.zeros([1, k]) # norm of each Xi

    Z = M.copy()
    Y = M.copy()
    eps = np.finfo(float).eps
    mu = 1/(np.mean(np.abs(W[:]))+eps)
    BBt = B @ D @ B.T

    for iter in range(1000):
        # update translation
        T = np.sum((W - Z @ B)@D, axis=1, keepdims=True) / (np.sum(np.diag(D)) + eps)
        W2fit = W - T

        # update motion matirx Z
        Z0 = Z.copy()
        Z = mrdivide(W2fit @ D @ B.T + mu * M + Y,
                     BBt + mu * np.eye(3*k))

        # update motion matrix M
        Q = Z - Y/mu
        for i in range(k):
            [M[:, 3*i:3*(i+1)], C[i]] = prox_2norm(Q[:, 3*i:3*(i+1)], alpha/mu)

        Y = Y + mu * (M-Z)

        PrimRes = np.linalg.norm(M-Z, 'fro') / (np.linalg.norm(Z0, 'fro') + eps)
        DualRes = mu * np.linalg.norm(Z-Z0, 'fro') / (np.linalg.norm(Z0, 'fro') + eps)

        if verb:
            print("Iter {}: PrimRes = {}, DualRes = {}, mu = {}".format(iter, PrimRes, DualRes, mu))


        if PrimRes < tol and DualRes < tol:
            break
        else:
            if PrimRes > 10 * DualRes:
                mu *= 2
            elif DualRes > 10 * PrimRes:
                mu /= 2

    [R, C] = syncRot(M)
    if np.sum(np.abs(R[:])) == 0:
        R = np.eye(3)

    R = R[0:2,:]
    S = np.kron(C, np.eye(3)) @ B

    fval = np.inf
    C = np.zeros([1, np.size(pc, 0)// 3])

    for iter in range(1000):
        T = np.sum((W - R @ S) @ D, 1, keepdims=True) / (np.sum(np.diag(D)) + eps)
        W2fit = W - T @ np.ones([1, p])

        R = estimateR_weighted(S, W2fit, D, R)

        if np.all(np.shape(pc) != 0):
            C0 = estimateC_weighted(W2fit, R, B, D, 1e-3)
            S = C0 * B
        else:
            C0 = estimateC_weighted(W2fit - R @ np.kron(C, np.eye(3)) @ pc, R, B, D, 1e-3)
            C = estimateC_weighted(W2fit - R @ C0 @ B, R, pc, D, lam)
            S = C0 * B + np.kron(C, np.eye(3)) @ pc
        fvaltml = fval
        fval = 0.5 * np.power(np.linalg.norm((W2fit - R @ S) @ np.sqrt(D), 'fro'), 2) + 0.5 * np.power(np.linalg.norm(C), 2)

        if verb:
            print("Iter: {}, fval = {}".format(iter, fval))

        if abs(fval-fvaltml)/fvaltml < tol:
            break

    R = np.append(R, [np.cross(R[0, :], R[1, :])], axis=0)

    return {
        "S": S,
        "M": M,
        "R": R,
        "C": C,
        "C0": C0,
        "T": T,
        "fval": fval
    }


def PoseFromKpts_FP(W, dict):
    pass


for idx in np.where(~annot.occluded & ~annot.truncated)[0]:
    id = idx+1
    imgname = annot.imgname[idx]
    center = annot.center[idx]
    scale = annot.scale[idx]
    clazz = getattr(annot, "class")[idx]
    indices = annot.indices[idx].astype(int) - 1
    cad_id = annot.cad_index[idx].astype(int) - 1

    cad = load("cad/{}.mat".format(clazz))
    cad = cad[clazz]
    cad = cad[cad_id]

    savefile = path.join(savepath, "valid_{}.mat".format(id))

    if cad_specific:
        d = getPascalTemplate(cad)
    else:
        d = load("dict/pca-{}.mat".format(clazz))

    valid_h5 = path.join(predpath, "valid_{}.h5".format(id))

    hm = h5py.File(valid_h5).get("heatmaps")
    hm = hm[indices[d["kpt_id"].astype(int) - 1],:,:]
    W_hp, score = findWmax(hm)

    output_wp = PoseFromKpts_WP(W_hp + 1, d, weight=score, verb=True, lam=1)
    pass