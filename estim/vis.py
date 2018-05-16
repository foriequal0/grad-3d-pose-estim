import matplotlib as mpl
import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize


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


def vis_fp(img, opt_fps, opt_wp, heatmap, center, scale, cad, path=None):
    # Some limiting conditions on Wedg
    from matplotlib.collections import PolyCollection

    img_crop = cropImage(img, center, scale)
    wplot = max(3, len(opt_fps))
    hplot = 2
    #h = plt.figure(None, (wplot, hplot), dpi=300)
    h, axes = plt.subplots(hplot, wplot, tight_layout=True)
    def add_axes(left, top):
        a = axes[top][left]
        a.set_axis_off()
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        return a
    a = add_axes(0, 0)
    a.imshow(img_crop)

    a = add_axes(1, 0)
    response = np.sum(heatmap, axis=0)
    max_value = np.max(response)
    jet = mpl.cm.get_cmap('jet')
    jet_img = jet(response/max_value)[:,:,0:3]
    jet_img = resize(jet_img, [200,200])
    a.imshow(jet_img * 0.5 + img_crop * 0.5)

    S_wp = opt_wp['R'] @ opt_wp['S'] + np.append(opt_wp['T'], [[0]], axis=0)
    model_wp, _, _, _ = fullShape(S_wp, cad)
    mesh2d_wp = model_wp["vertices"][:, 0:2] * 200 / np.size(heatmap, 1)

    a = add_axes(2, 0)
    a.imshow(img_crop)
    a.hold()
    a.add_collection(
        PolyCollection(
            [mesh2d_wp[face] for face in model_wp["faces"].astype(int)-1],
            alpha=0.4)
    )

    i=0
    for opt_fp in opt_fps:
        K = opt_fp["K"]
        S_fp = opt_fp["R"] @ opt_fp["S"] + opt_fp["T"]
        model_fp, w, _, T_fp = fullShape(S_fp, cad)

        S_fp = opt_fp["R"] @ opt_fp["S"] + opt_fp["T"]
        model_fp, _, _, _ = fullShape(S_fp, cad)
        mesh2d_fp = K @ model_fp["vertices"].T
        mesh2d_fp = (mesh2d_fp[0:2, :] / mesh2d_fp[2,:])
        mesh2d_fp = transformHG(mesh2d_fp, center, scale, heatmap[0].shape, False) *200 / np.size(heatmap, 1)
        mesh2d_fp = mesh2d_fp.T
        a = add_axes(i, 1)
        a.imshow(img_crop)
        a.hold()
        a.add_collection(
            PolyCollection(
                [mesh2d_fp[face] for face in model_fp["faces"].astype(int)-1],
                alpha=0.4)
        )
        i+=1

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(h)