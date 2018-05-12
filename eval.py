import scipy.io as sio
import os.path as path
import h5py
import numpy as np
from object3d.opts import opts
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions
from pymanopt import Problem
from imageio import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
mpl.use('WXAgg')

def load(path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False, mat_dtype=True)


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
        if np.prod(np.shape(xyz)) != 0:
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

def mldivide(a, b):
    # return np.linalg.pinv(a) @ b
    return np.linalg.lstsq(a, b)

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

def composeShape(B, C):
    if np.size(C, 1) != np.size(B, 0) // 3:
        C = C.T

    f = np.size(C, 0)
    p = np.size(B, 1)
    k = np.size(B, 0) // 3

    B = np.reshape(B.T, [3 * p, k])
    S = np.reshape(B @ C.T, [p, 3*f])
    return S


def PoseFromKpts_FP(W, d, lam=1, tol=1e-3, weight=None, r0=None, verb=True):
    D = np.eye(np.size(W, 1)) if weight is None else np.diag(weight)
    R = np.eye(3) if r0 is None else r0

    mu = centralize(d["mu"])
    pc = centralize(d["pc"])

    eps = np.finfo(float).eps

    S = mu.copy()
    T = np.mean(W, 1, keepdims=True) * np.mean(np.std(R[0:2, :] @ S, axis=1, keepdims=True)) / np.mean(np.std(W, axis=1) + eps)
    C = 0

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
        "Z": Z
    }

def get_transform(center, scale, res):
    h = 200 * scale
    t = np.eye(3, 3)
    t[0, 0] = res[1]/h
    t[1, 1] = res[0]/h
    t[0, 2] = res[1] * (-center[0]/h + 0.5)
    t[1, 2] = res[0] * (-center[1] / h + 0.5)
    t[2, 2] = 1
    return t

def transformHG(pt, center, scale, res, invert):
    t = get_transform(center, scale, res)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.stack([pt[0,:], pt[1,:]-0.5, np.ones([np.size(pt, 1)])])
    new_pt = t @ new_pt
    return new_pt[0:2, :]


def findRotation(S1, S2):
    F, P = np.shape(S1)
    F = F // 3
    S1 = np.reshape(S1, [3, F * P])
    S2 = np.reshape(S2, [3, F * P])
    R = S1 @ S2.T
    [U, _, V] = np.linalg.svd(R)
    R = U @ V
    R = U @ np.diag([1,1,np.linalg.det(R)]) @ V
    return R


def fullShape(S1, model, kpt2fit=None):
    if kpt2fit is None:
        kpt2fit = list(range(np.size(S1, 1)))

    S2 = np.zeros([3, len(kpt2fit)])
    for i in range(len(kpt2fit)):
        xyz = model[model["pnames"][kpt2fit[i]]]
        S2[:, i] = xyz.T

    eps = np.finfo(float).eps
    T1 = np.mean(S1, axis=1, keepdims=True)
    S1 = S1 - T1
    T2 = np.mean(S2, axis=1, keepdims=True)
    S2 = S2 - T2
    R = findRotation(S1, S2)
    S2 = R @ S2
    w = np.trace(S1.T @ S2) / (np.trace(S2.T @ S2) + eps)
    T = T1 - w * R @ T2
    vertices = w * R @ (model["vertices"].T - T2) + T1
    model_new = model.copy()
    model_new["vertices"] = vertices.T
    return model_new, w, R, T


def cropImage(im, center, scale):
    w = int(200*scale)
    h = int(w)
    x = int(center[0] - w/2)
    y = int(center[1] - h/2)
    im = np.pad(im, [(h, h), (w, w), (0,0)], 'constant', constant_values=0)
    lt = np.array([y, x]) + np.array([h, w])
    rb = lt + np.array([w, h])
    im1 = im[lt[0]:rb[0], lt[1]:rb[1], :]
    im1 = resize(im1, [200, 200])
    return im1

def vis_wp(img, opt, heatmap, center, scale, cad, d):
    # Some limiting conditions on Wedg
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    img_crop = cropImage(img, center, scale)
    nplot = 3
    h = plt.figure(None, (nplot, 1))

    def add_axes(left):
        a = h.add_axes(((left - nplot/2+0.5)/nplot, 0, 1, 1))
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        return a
    a = add_axes(0)
    a.imshow(img_crop)

    a = add_axes(1)
    response = np.sum(heatmap, axis=0)
    max_value = np.max(response)
    jet = mpl.cm.get_cmap('jet')
    jet_img = jet(response/max_value)[:,:,0:3]
    jet_img = resize(jet_img, [200,200])
    a.imshow(jet_img * 0.5 + img_crop * 0.5)

    S_wp = opt['R'] @ opt['S'] + np.append(opt['T'], [[0]], axis=0)
    model, _, _, _ = fullShape(S_wp, cad)

    faces = model["faces"].astype(int) - 1
    vertices = model["vertices"]

    mesh2d = vertices[:, 0:2] * 200 / np.size(heatmap, 1)

    a = add_axes(2)
    a.imshow(img_crop)
    a.hold()
    a.add_collection(
        PolyCollection(
            [mesh2d[face] for face in faces],
            alpha=0.4)
    )

    plt.show()

def vis_fp(img, opt_fp, opt_wp, heatmap, center, scale, K, cad):
    # Some limiting conditions on Wedg
    from matplotlib.collections import PolyCollection

    img_crop = cropImage(img, center, scale)
    nplot = 4
    h = plt.figure(None, (nplot, 1))

    def add_axes(left):
        a = h.add_axes(((left - nplot/2+0.5)/nplot, 0, 1, 1))
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        return a
    a = add_axes(0)
    a.imshow(img_crop)

    a = add_axes(1)
    response = np.sum(heatmap, axis=0)
    max_value = np.max(response)
    jet = mpl.cm.get_cmap('jet')
    jet_img = jet(response/max_value)[:,:,0:3]
    jet_img = resize(jet_img, [200,200])
    a.imshow(jet_img * 0.5 + img_crop * 0.5)

    S_wp = opt_wp['R'] @ opt_wp['S'] + np.append(opt_wp['T'], [[0]], axis=0)
    model_wp, _, _, _ = fullShape(S_wp, cad)
    mesh2d_wp = model_wp["vertices"][:, 0:2] * 200 / np.size(heatmap, 1)

    a = add_axes(2)
    a.imshow(img_crop)
    a.hold()
    a.add_collection(
        PolyCollection(
            [mesh2d_wp[face] for face in model_wp["faces"].astype(int)-1],
            alpha=0.4)
    )

    S_fp = opt_fp["R"] @ opt_fp["S"] + opt_fp["T"]
    model_fp, _, _, _ = fullShape(S_fp, cad)
    mesh2d_fp = K @ model_fp["vertices"].T
    mesh2d_fp = (mesh2d_fp[0:2, :] / mesh2d_fp[2,:])
    mesh2d_fp = transformHG(mesh2d_fp, center, scale, heatmap[0].shape, False) *200 / np.size(heatmap, 1)
    mesh2d_fp = mesh2d_fp.T
    a = add_axes(3)
    a.imshow(img_crop)
    a.hold()
    a.add_collection(
        PolyCollection(
            [mesh2d_fp[face] for face in model_fp["faces"].astype(int)-1],
            alpha=0.4)
    )

    plt.show()

def demo_fp():
    datapath = 'demo/gascan/'
    d = load(path.join(datapath, 'annot/dict.mat'))
    cad = load(path.join(datapath, 'annot/cad.mat'))
    annotfile = path.join(datapath, 'annot/valid.mat')
    annotmat = load(annotfile)
    annot = annotmat["annot"]

    for idx in range(len(annot.imgname)):
        ID = idx +1
        imgname = annot.imgname[idx]
        center = annot.center[idx,:]
        scale = annot.scale[idx]
        K = annot.K[idx]

        valid_h5 = path.join(datapath, "exp/valid_{}.h5".format(ID))
        heatmap = h5py.File(valid_h5).get("heatmaps")[0]
        W_hp, score = findWmax(heatmap)
        W_hp = W_hp
        W_im = transformHG(W_hp, center, scale, heatmap.shape[1:], True)
        W_ho = mldivide(K, np.concatenate([W_im, np.ones([1, np.size(W_im, 1)])]))[0]

        output_wp = PoseFromKpts_WP(W_hp, d, weight=score, verb=False)

        output_fp = PoseFromKpts_FP(W_ho, d, r0=output_wp["R"], weight=score, verb=True)
        S_fp = output_fp["R"] @ output_fp["S"] + output_fp["T"]
        model_fp, w, _, T_fp = fullShape(S_fp, cad)
        output_fp["T_metric"] = T_fp/w

        img = imread(path.join(datapath, "images/{}".format(imgname)))
        vis_fp(img, output_fp, output_wp, heatmap, center, scale, K, cad)

demo_fp()
exit(0)

def pascal3d_eval():
    datapath = path.join(opts.data_dir, "pascal3d/annot")
    predpath = 'exp/pascal3d/'
    savepath = 'result/cad/'

    cad_specific = False

    annotfile = path.join(datapath, "valid.mat")
    annotmat = load(annotfile)
    annot = annotmat["annot"]

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

        img = imread(path.join(datapath, "../images/{}.jpg".format(imgname)))
        vis_wp(img, output_wp, hm, center, scale, cad.__dict__, d)

pascal3d_eval()