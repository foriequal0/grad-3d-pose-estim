from os import path as path

import h5py
import numpy as np
from imageio import imread

from estim.util import load, findWmax
from .vis import transformHG, fullShape, vis_fp, vis_wp
from .core import PoseFromKpts_WP, PoseFromKpts_FP, PoseFromKpts_FP_estim_K_using_WP, PoseFromKpts_FP_estim_K_solely
from .util import mldivide
from object3d.opts import opts


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
        W_im = transformHG(W_hp, center, scale, heatmap.shape[1:], True)

        def reproj_err(x):
            err = W_hp - x
            return np.mean(np.sqrt(np.diag(err.T @ err)) * score)

        def reproj_fp(output, K):
            fp = K @ (output["R"] @ output["S"] + output["T"])
            p = fp[0:2] / fp[2]
            return reproj_err(transformHG(p, center, scale, heatmap.shape[1:], False))

        output_wp = PoseFromKpts_WP(W_hp, d, weight=score, verb=False)
        wp_reproj = reproj_err((output_wp["R"] @ output_wp["S"])[0:2] + output_wp["T"])
        print("err wp: ", wp_reproj)

        W_ho = mldivide(K, np.concatenate([W_im, np.ones([1, np.size(W_im, 1)])]))[0]
        output_fp = PoseFromKpts_FP(W_ho, d, r0=output_wp["R"], weight=score, verb=False)
        output_fp["K"] = K
        fp_reproj = reproj_fp(output_fp, K)
        print("err fp: ", fp_reproj)
        print("fval FP: ", output_fp["fval"])

        output_fp2 = PoseFromKpts_FP_estim_K_using_WP(W_im, d, output_wp, score, center, scale, heatmap.shape[1])
        fp2_reproj = reproj_fp(output_fp2, output_fp2["K"])
        print("err fp2: ", fp2_reproj)
        print("fval FP_estim_K_using_WP: ", output_fp2["fval"], output_fp2["f"], output_fp2["d"])

        output_fp3 = PoseFromKpts_FP_estim_K_solely(W_im, d, r0=output_wp["R"], weight=score, verb=False)
        fp3_reproj = reproj_fp(output_fp3, output_fp3["K"])
        print("err fp3: ", fp3_reproj)
        print("fval FP_estim_K_solely: ", output_fp3["fval"],
              output_fp3["f"],
              output_fp3["d"]
        )

        img = imread(path.join(datapath, "images/{}".format(imgname)))
        vis_fp(img, [output_fp, output_fp2, output_fp3], output_wp, heatmap, center, scale, K, cad)


def pascal3d_eval():
    datapath = path.join(opts.data_dir, "pascal3d/annot")
    predpath = 'exp/pascal3d/'
    savepath = 'result/cad/'

    cad_specific = False

    annotfile = path.join(datapath, "valid.mat")
    annotmat = load(annotfile)
    annot = annotmat["annot"]

    for idx in np.where(~annot.occluded & ~annot.truncated & (getattr(annot, "class") != "aeroplane"))[0]:
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

        output_wp = PoseFromKpts_WP(W_hp + 1, d, weight=score, verb=False, lam=1)
        img = imread(path.join(datapath, "../images/{}.jpg".format(imgname)))
        vis_wp(img, output_wp, hm, center, scale, cad.__dict__, d)