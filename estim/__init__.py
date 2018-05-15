import numpy as np

from .util import mrdivide, centralize, reshapeS
from . import core, util


indices_of_class = {
    'aeroplane': range(8),
    'bicycle': range(8, 19),
    'boat': range(19, 26),
    'bottle': range(26, 33),
    'bus': range(33, 45),
    'car': range(45, 57),
    'chair': range(57, 67),
    'sofa': range(67, 77),
    'train': range(77, 94),
    'tvmonitor': range(94, 33),
}

def estim_wp(hm, clazz):
    d = util.load("dict/pca-{}.mat".format(clazz))
    indices = np.array(list(indices_of_class[clazz])),
    hm = hm[indices[d["kpt_id"].astype(int) - 1], :, :]
    W_hp, score = util.findWmax(hm)

    output_wp = core.PoseFromKpts_WP(W_hp + 1, d, weight=score, verb=False, lam=1)
    return output_wp


def estim_fp(hm, clazz):
    d = util.load("dict/pca-{}.mat".format(clazz))
    indices = np.array(list(indices_of_class[clazz])),
    hm = hm[indices[d["kpt_id"].astype(int) - 1], :, :]
    W_hp, score = util.findWmax(hm)

    output_wp = core.PoseFromKpts_WP(W_hp + 1, d, weight=score, verb=False, lam=1)
    return output_wp