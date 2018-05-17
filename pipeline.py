import object3d.idle_gpu as idle_gpu
try:
    idle_gpu.try_limit()
except idle_gpu.GPUStatNotFoundError:
    pass

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import os.path as path
import h5py

import estim
import object3d.model as model
import object3d.main as main
import object3d.loader as loader
from object3d.opts import opts

import numpy as np

def process():
    import pathlib
    from imageio import imread
    from estim.util import load, findWmax
    from estim.vis import transformHG,  vis_fp
    from estim.core import PoseFromKpts_WP,  PoseFromKpts_FP_estim_K_using_WP, PoseFromKpts_FP_estim_K_solely
    from object3d.opts import opts
    import csv

    datapath = path.join(opts.data_dir, "pascal3d/annot")
    predpath = 'exp/pascal3d/'

    annotfile = path.join(datapath, "valid.mat")
    annotmat = load(annotfile)
    annot = annotmat["annot"]

    pathlib.Path("plot").mkdir(parents=True, exist_ok=True)

    csvfile = open('result.csv', 'w', newline='')
    csvwriter = csv.DictWriter(
        csvfile,
        fieldnames=["id", "class", "hm_err",
                    "nonzeros",
                    "score sum",
                    "score > 0.1",

                    "hm_wp_reproj_err",
                    "hm_fp2_reproj_err",
                    "hm_fp3_reproj_err",

                    "hm_wp_reproj_err_to_gt",
                    "hm_fp2_reproj_err_to_gt",
                    "hm_fp3_reproj_err_to_gt",
                    ])

    def do(id, hm_gt, hm):
        idx = id - 1
        imgname = annot.imgname[idx]
        center = annot.center[idx]
        scale = annot.scale[idx]
        clazz = getattr(annot, "class")[idx]
        indices = annot.indices[idx].astype(int) - 1
        cad_id = annot.cad_index[idx].astype(int) - 1

        cad = load("cad/{}.mat".format(clazz))
        cad = cad[clazz]
        cad = cad[cad_id]

        d = load("dict/pca-{}.mat".format(clazz))

        valid_h5 = path.join(predpath, "valid_{}.h5".format(id))

        hm = np.transpose(hm, [2, 0, 1])
        hm = hm[indices[d["kpt_id"].astype(int) - 1], :, :]

        # hm_after_count = np.sum(np.max(np.max(hm, axis=1), axis=1) > 0.1)
        # if hm_after_count < 4:
        #     return
        hm_gt = np.transpose(hm_gt, [2, 0, 1])
        hm_gt = hm_gt[indices[d["kpt_id"].astype(int) - 1], :, :]

        W_hp, score = findWmax(hm)
        W_hp_gt, score_gt = findWmax(hm_gt)
        score_gt = score_gt > 0

        def reproj_err(W, x, score):
            err = W - x
            return np.sum(np.sqrt(np.diag(err.T @ err)) * np.power(score, 1/2) ) / np.sum(score>0)

        def reproj_fp(W, output, K, score):
            fp = K @ (output["R"] @ output["S"] + output["T"])
            p = fp[0:2] / fp[2]
            return reproj_err(W, transformHG(p, center, scale, hm.shape[1:], False), score)

        mirrors = [ np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]), np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]), np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]), np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]), np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]), np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
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

        csvwriter.writerow({
            "id": id,
            "class": clazz,
            "nonzeros": np.sum(score_gt),
            "score sum": np.sum(score),
            "score > 0.1": np.sum(score > 0.1),
            "hm_err": reproj_err(W_hp_gt, W_hp, score_gt),

            "hm_wp_reproj_err":  reproj_err(W_hp, (wp["R"] @ wp["S"])[0:2] + wp["T"], score),
            "hm_fp2_reproj_err": fp2_err,
            "hm_fp3_reproj_err": fp3_err,

            "hm_wp_reproj_err_to_gt":  reproj_err(W_hp_gt, (wp["R"] @ wp["S"])[0:2] + wp["T"], score_gt),
            "hm_fp2_reproj_err_to_gt": reproj_fp(W_hp_gt, fp2, fp2["K"], score_gt),
            "hm_fp3_reproj_err_to_gt": reproj_fp(W_hp_gt, fp3, fp3["K"], score_gt),
        })

        img = imread(path.join(datapath, "../images/{}.jpg".format(imgname)))
        vis_fp(img, [fp2, fp3], wp, hm, center, scale, cad.__dict__, "plot/{}.jpg".format(id))
        csvfile.flush()

    return do

def load(label):
    names_filename = path.join(opts.data_dir, "pascal3d", "annot",
                               "{}_images.txt".format(label))

    images = []

    with open(names_filename) as names_file:
        lines = names_file.read().splitlines()
        for index, line in enumerate(lines):
            line = line.rstrip()
            images.append(line)

    annot = h5py.File(path.join(opts.data_dir, "pascal3d", "annot", "{}.h5".format(label)))

    for i in range(len(annot["index"])):
        yield {
            "image": images[i],
            "index": annot["index"][i],
            "part": annot["part"][i],
            "center": annot["center"][i],
            "scale": annot["scale"][i]
        }


input = tf.placeholder(tf.float32, (None, 256, 256, 3))
out1, out2 = model.stacked_hourglass(input, 102, False)

with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("./model/default")
    saver.restore(sess, checkpoint)

    p = process()

    for i in load("valid"):
        if i["index"] % 10 != 0:
            continue

        print(i["index"])

        i = sess.run(loader.make_input_and_labels({k:tf.convert_to_tensor(v) for k,v in i.items() }))

        hm_pre_count = np.sum(np.max(np.max(i["labels"], axis=0), axis=0) > 0)
        # if hm_pre_count < 4:
        #     continue
        res, = sess.run(out1, {
            input: np.expand_dims(i["input"], axis=0)
        })
        try:
            p(i["index"], i["labels"], res)
        except Exception as e:
            print(e)

