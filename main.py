import object3d.idle_gpu as idle_gpu
try:
    idle_gpu.try_limit()
except idle_gpu.GPUStatNotFoundError:
    pass

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

import object3d.main as main
from object3d.opts import opts

if __name__ == "__main__":
    if opts.mode == "train":
        main.train_main()
    elif opts.mode == "pred":
        main.pred_main()
    elif opts.mode == "dummy":
        main.dummy_main()
    else:
        raise ValueError("Invalid mode " + opts.mode)

