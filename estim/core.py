import numpy as np
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions

from .util import mldivide, mrdivide, centralize, reshapeS
from .linesearch import linesearch, linesearch1d, ChangeDetect

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

    def cost(X):
        E = A @ X - B
        f = np.trace(E.T @ D @ E) / (2*N)
        return f

    def grad(X):
        E = A @ X - B
        egrad = (At /N) @ (D @ E)
        g = manifold.egrad2rgrad(X, egrad)
        return g

    manifold = Stiefel(n, p)
    problem = Problem(manifold, cost=cost, grad=grad, verbosity=0)
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

    y = np.expand_dims(W.flatten('F'), 1)
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

        if np.prod(np.shape(pc)) == 0:
            C0 = estimateC_weighted(W2fit, R, B, D, 1e-3)
            S = C0 * B
        else:
            C0 = estimateC_weighted(W2fit - R @ np.kron(C, np.eye(3)) @ pc, R, B, D, 1e-3)
            C = estimateC_weighted(W2fit - R * C0 @ B, R, pc, D, lam)
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


def composeShape(B, C):
    if np.size(C, 1) != np.size(B, 0) // 3:
        C = C.T

    f = np.size(C, 0)
    p = np.size(B, 1)
    k = np.size(B, 0) // 3

    B = np.reshape(B.T, [3 * p, k])
    S = np.reshape(B @ C.T, [p, 3*f])
    return S.T


def PoseFromKpts_FP(W, d, lam=1, tol=1e-3, weight=None, r0=None, verb=True):
    D = np.eye(np.size(W, 1)) if weight is None else np.diag(weight)
    R = np.eye(3) if r0 is None else r0

    mu = centralize(d["mu"])
    pc = centralize(d["pc"])

    eps = np.finfo(float).eps

    S = mu.copy()
    T = np.mean(W, 1, keepdims=True) * np.mean(np.std(R[0:2, :] @ S, axis=1, keepdims=True)) / np.mean(np.std(W, axis=1) + eps)
    C = np.zeros([1, np.size(pc, 0)//3])

    fval = np.inf
    for iter in range(1000):
        Z = np.sum(W * ((R@S) + T), axis=0) / (np.sum(W ** 2, axis=0) + eps)

        Sp = W @ np.diag(Z)
        T = np.sum((Sp - R@S)@D, axis=1, keepdims=True)/(np.sum(np.diag(D)) + eps)
        St = Sp - T
        U,_,V = np.linalg.svd(St@D@S.T)

        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U@V.T))]) @ V

        if np.prod(np.size(pc)) != 0:
            y = reshapeS(R.T @ St - mu, 'b2v')
            X = reshapeS(pc, 'b2v')
            C = np.linalg.pinv(X.T @ X + lam * np.eye(np.max(np.shape(C)))) @ X.T @ y
            S = mu + composeShape(pc, C)

        fvaltml = fval
        fval = np.linalg.norm((St - R@S) @ np.sqrt(D), 'fro') ** 2 + lam * np.linalg.norm(C) ** 2

        if verb:
            print("Iter: {}, fval = {}".format(iter, fval))

        if abs(fval - fvaltml) / (fvaltml + eps) < tol:
            break

    return {
        "S": S,
        "R": R,
        "C": C,
        "T": T,
        "Z": Z,
        "fval": fval
    }


def rotationMatrix(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    skew_sym = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + skew_sym + skew_sym@skew_sym/(1+c)


def normalize(x):
    mean = np.mean(x, 1, keepdims=True)
    std = np.std(x, 1, keepdims=True)
    return (x-mean)/std

def average(X, axis=None, weights=1):
    return np.sum(X * weights, axis) / (np.sum(weights) + eps)

def centralize_points(x, weight=1):
    mean = average(x, 1, weight)
    return (x-np.expand_dims(mean, 1))

def PoseFromKpts_FP_estim_K_using_WP(W_im, dict, output_wp, score, center, scale, res):
    R = output_wp["R"]
    S = output_wp["S"] / res * (200 * scale)
    T = (output_wp["T"][:,0] / res - 0.5) * (200 * scale) + center

    normalized_W_im = normalize(W_im)

    def foverr(f, d):
        h = T - d
        r = rotationMatrix(np.array([0, 0, f]), np.array([h[0], h[1], f]))
        ryz = r @ R @ S
        p = (ryz[0:2] + np.expand_dims(h, 1)) / (ryz[2] + f) * f + np.expand_dims(d, 1)

        return np.sum((((centralize_points(W_im, score) - centralize_points(p, score))) ** 2) * score)

    f0 = 1000.0
    dx0 = 320.0
    dy0 = 240.0
    f, dx, dy = linesearch(lambda x: foverr(x[0], x[1:3]),
                           np.array([f0, dx0, dy0]),  100.0,
                           np.array([10, 0, 0]),
                           np.array([100000, 1000, 1000]),
                           100)

    K = np.array([
        [f, 0, dx],
        [0, f, dy],
        [0, 0, 1],
    ])
    W_ho = mldivide(K, np.concatenate([W_im, np.ones([1, np.size(W_im, 1)])]))[0]
    output_fp2 = PoseFromKpts_FP(W_ho, dict, r0=output_wp["R"], weight=score, verb=False)
    output_fp2["f"] = f
    output_fp2["d"] = [dx,dy]
    output_fp2["K"] = K
    return output_fp2


eps = np.finfo(float).eps


def estimSize(X, weights=1):
    center = average(X, 1, weights=weights)
    centered = X - np.expand_dims(center, 1)

    distance = np.sqrt(np.diag(centered.T @ centered))
    mean = average(distance, weights=weights)
    var = average((distance-mean)**2, weights=weights)
    return np.sqrt(var)

def PoseFromKpts_FP_estim_K_solely(W_im, dict, lam=1, tol=1e-3, weight=None, r0=None, verb=True):
    D = np.eye(np.size(W_im, 1)) if weight is None else np.diag(weight)
    R = np.eye(3) if r0 is None else r0

    mu = centralize(dict["mu"])
    pc = centralize(dict["pc"])

    S = mu.copy()
    normalized_W_im = normalize(W_im)
    RS = R @ S
    def foverr(f, di):
        t0 = np.mean(W_im, 1) - di
        r = rotationMatrix(np.array([0,0,f]), np.array([t0[0], t0[1], f]))
        rRS = r@RS
        k0 = f * estimSize(RS[0:2]) / estimSize(W_im)
        def err(t, k):
            p = (rRS[0:2] + np.expand_dims(t, 1) / f * k) / (rRS[2] + k) + np.expand_dims(di, 1)
            return np.sum(((centralize_points(W_im, weight) - centralize_points(p, weight)) ** 2) * weight)
        return err(t0, k0)


    f0 = 1000.0
    dx0 = 320.0
    dy0 = 240.0
    f, dx, dy = linesearch(lambda x: foverr(x[0], x[1:3]),
                           np.array([f0, dx0, dy0]), 100.0,
                           np.array([10, 0, 0]),
                           np.array([100000, 1000, 1000]),
                           100)

    K = np.array([
        [f, 0, dx],
        [0, f, dy],
        [0, 0, 1],
    ])

    W = mldivide(K, np.concatenate([W_im, np.ones([1, np.size(W_im, 1)])]))[0]

    T = np.mean(W, 1, keepdims=True) * np.mean(np.std(R[0:2, :] @ S, axis=1, keepdims=True)) / np.mean(
        np.std(W, axis=1) + eps)
    C = np.zeros([1, np.size(pc, 0) // 3])

    fval = np.inf
    for iter in range(1000):
        Z = np.sum(W * ((R@S) + T), axis=0) / (np.sum(W ** 2, axis=0) + eps)

        Sp = W @ np.diag(Z)
        T = np.sum((Sp - R@S)@D, axis=1, keepdims=True)/(np.sum(np.diag(D)) + eps)
        St = Sp - T
        U,_,V = np.linalg.svd(St@D@S.T)

        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U@V.T))]) @ V

        if np.prod(np.size(pc)) != 0:
            y = reshapeS(R.T @ St - mu, 'b2v')
            X = reshapeS(pc, 'b2v')
            C = np.linalg.pinv(X.T @ X + lam * np.eye(np.max(np.shape(C)))) @ X.T @ y
            S = mu + composeShape(pc, C)

        fvaltml = fval
        fval = np.linalg.norm((St - R@S) @ np.sqrt(D), 'fro') ** 2 + lam * np.linalg.norm(C) ** 2

        if verb:
            print("Iter: {}, fval = {}".format(iter, fval))

        if abs(fval - fvaltml) / (fvaltml + eps) < tol:
            break

    return {
        "S": S,
        "R": R,
        "C": C,
        "T": T,
        "Z": Z,
        "fval": fval,
        "f": f,
        "d": [dx,dy],
        "K": K,
    }