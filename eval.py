import tensorflow as tf
import object3d.model as model
import object3d.loader as loader
import estim.util as util
import numpy as np
import os.path as path
from estim.demo import pascal3d_eval, demo_fp

from imageio import imread
from estim.util import load, findWmax
from estim.vis import transformHG, vis_fp
from estim.core import PoseFromKpts_WP, PoseFromKpts_FP_estim_K_using_WP, PoseFromKpts_FP_estim_K_solely
from object3d.opts import opts

input = tf.placeholder(tf.float32, (None, 256, 256, 3))
out1, out2 = model.stacked_hourglass(input, 102, False)

with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("./model/default")
    saver.restore(sess, checkpoint)

    imgname = "./snucar.jpg"
    center= np.array([357,274])
    scale = 2.5
    clazz = "car"
    kpts = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

    i = sess.run(loader.make_input_fullpath({
        "image": tf.constant(imgname),
        "center": tf.constant(center),
        "scale": tf.constant(2.5),
    }))

    hm, = sess.run(out1, {
        input: np.expand_dims(i["input"], axis=0)
    })

    cad = util.load("cad/{}.mat".format(clazz))
    cad = cad[clazz]
    cad = cad[1]

    d = util.load("dict/pca-{}.mat".format(clazz))

    hm = np.transpose(hm, [2, 0, 1])
    hm = hm[kpts, :, :]

    W_hp, score = findWmax(hm)

    def reproj_err(W, x, score):
        err = W - x
        return np.sum(np.sqrt(np.diag(err.T @ err)) * np.power(score, 1 / 2)) / np.sum(score > 0)


    def reproj_fp(W, output, K, score):
        fp = K @ (output["R"] @ output["S"] + output["T"])
        p = fp[0:2] / fp[2]
        return reproj_err(W, transformHG(p, center, scale, hm.shape[1:], False), score)

    mirrors = [np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]), np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]), np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]), np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ]), np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ]), np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])]

    W_im = transformHG(W_hp, center, scale, hm.shape[1:], True)

    wp = PoseFromKpts_WP(W_hp + 1, d, weight=score, verb=False, lam=1)
    fp2 = PoseFromKpts_FP_estim_K_using_WP(W_im, d, wp, score, center, scale, hm.shape[1])
    fp2_err = reproj_fp(W_hp, fp2, fp2["K"], score)
    for mirror in mirrors:
        wp_alt = wp.copy()
        wp_alt["R"] = wp["R"] @ mirror
        a = PoseFromKpts_FP_estim_K_using_WP(W_im, d, wp_alt, score, center, scale, hm.shape[1])
        err = reproj_fp(W_hp, a, a["K"], score)
        if err < fp2_err:
            fp2 = a
            fp2_err = err

    fp3 = PoseFromKpts_FP_estim_K_solely(W_im, d, r0=wp["R"], weight=score, verb=False)
    fp3_err = reproj_fp(W_hp, fp3, fp3["K"], score)
    for mirror in mirrors:
        a = PoseFromKpts_FP_estim_K_solely(W_im, d, r0=wp["R"] @ mirror, weight=score, verb=False)
        err = reproj_fp(W_hp, a, a["K"], score)
        if err < fp3_err:
            fp3 = a
            fp3_err = err

    img = imread(imgname)
    vis_fp(img, [fp2, fp3], wp, hm, center, scale, cad.__dict__)
    pass