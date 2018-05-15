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

input = tf.placeholder(tf.float32, (None, 256, 256, 3))
out1, out2 = model.stacked_hourglass(input, 102, False)


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
            "part": annot["part"][i],
            "center": annot["center"][i],
            "scale": annot["scale"][i]
        }


with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("./model/default")
    saver.restore(sess, checkpoint)

    for i in load("valid"):
        i = {k:tf.convert_to_tensor(v) for k,v in i.items() }
        t = sess.run(loader.make_input(i))
        res, = sess.run(out1, {
            input: np.expand_dims(t["input"], axis=0)
        })

        estim.estim_fp()

        pass