def get_idle_gpu():
    import subprocess
    import json
    gpustat_json = subprocess.check_output("gpustat --json", shell=True)
    gpustat = json.loads(gpustat_json.decode("utf-8"))
    for gpu in gpustat["gpus"]:
        if len(gpu["processes"]) == 0:
            return int(gpu["index"])

    return None


def cuda_visible_to_idle():
    import os

    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        return

    import shutil
    if not shutil.which("gpustat"):
        return

    idle_gpu = get_idle_gpu()
    if idle_gpu is None:
        raise Exception("No idle gpu")

    os.putenv("CUDA_VISIBLE_DEVICES", str(idle_gpu))

    print("CUDA_VISIBLE_DEVICES:", idle_gpu, "check:", os.getenv("CUDA_VISIBLE_DEVICES"))


cuda_visible_to_idle()

from os import path
import h5py
import tensorflow as tf
from stacked_hourglass.opts import opts

tf.logging.set_verbosity(tf.logging.DEBUG)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def to_channel_first(x, batch=True):
    if batch:
        return tf.transpose(x, perm=[0, 3, 1, 2])
    else:
        return tf.transpose(x, perm=[2, 0, 1])


def to_channel_last(x, batch=True):
    if batch:
        return tf.transpose(x, perm=[0, 2, 3, 1])
    else:
        return tf.transpose(x, perm=[1, 2, 0])


def load_annots(label):
    names_filename = path.join(opts.data_dir, "pascal3d", "annot",
                               "{}_images.txt".format(label))

    images = []
    image_to_indices = {}

    with open(names_filename) as names_file:
        lines = names_file.read().splitlines()
        for index, line in enumerate(lines):
            line = line.rstrip()
            images.append(line)

            indices = image_to_indices.setdefault(line, [])
            indices.append(index)

    annot = h5py.File(path.join(opts.data_dir, "pascal3d", "annot", "{}.h5".format(label)))

    return {
        "dataset": tf.data.Dataset.from_tensor_slices({
            "image": images,
            "part": annot["part"],
            "center": annot["center"],
            "scale": annot["scale"],
            "index": annot["index"],
        }),
        "ref": {
            "nsamples": annot["part"].shape[0],
            "nparts": annot["part"].shape[1],
            "image_to_indices": image_to_indices,
        },
    }


def crop(image, center, scale):
    input_res = tf.to_float(opts.input_res)
    scale = tf.to_float(scale)
    width = tf.to_int32(input_res * scale)
    top_left = tf.to_int32(center - input_res * scale / 2) + width  # to be padded

    input_channel = image.get_shape()[2]
    padded = tf.pad(image, [[width, width], [width, width], [0, 0]], "CONSTANT")
    padded.set_shape([None, None, input_channel])  # static channel is lost while padding

    cropped = tf.image.crop_to_bounding_box(padded, top_left[1], top_left[0], width, width)

    resized = tf.image.resize_images(cropped, [opts.input_res, opts.input_res])

    return resized


def draw_gaussian(pos, center, scale, sigma):
    # TODO: opts invariant

    input_res = tf.to_float(opts.input_res)
    output_res = tf.to_float(opts.output_res)

    center = tf.to_float(center)
    sigma = tf.to_float(sigma)
    scale = tf.to_float(scale)
    top_left = center - input_res * scale / 2

    c = (pos - top_left) / scale / input_res * output_res

    range = tf.expand_dims(tf.range(output_res), 1)
    y = tf.transpose(range) - c[0]
    x = range - c[1]
    X = tf.exp(-(tf.pow(x, 2) + tf.pow(y, 2)) / (2 * tf.pow(sigma, 2)))
    return X


def make_input_crop(annot):
    filename = annot["image"]
    image_path = tf.string_join([opts.data_dir, "pascal3d", "images", filename], separator="/")
    center = tf.to_float(annot["center"])
    scale = tf.to_float(annot["scale"]) / 1.28  # 1.28 is 256/200, original paper impl.'s flaw

    image_string = tf.read_file(image_path)
    image_decoded = tf.cast(tf.image.decode_image(image_string, channels=3), tf.float32) / 255.0
    image_decoded.set_shape([None, None, 3])  # decode_image doesn't set channels correctly

    image_cropped = crop(image_decoded, center, scale)

    result = annot.copy()
    result["input"] = image_cropped
    return result


def make_feature_and_labels(annot):
    annot = make_input_crop(annot)

    parts = tf.to_float(annot["part"])
    center = tf.to_float(annot["center"])
    scale = tf.to_float(annot["scale"]) / 1.28  # 1.28 is 256/200, original paper impl.'s flaw

    def make_labels(part):
        return tf.cond(
            tf.greater(part[1], 0),
            true_fn=lambda: draw_gaussian(part, center, scale, 1.75),
            false_fn=lambda: tf.zeros([64, 64], tf.float32))

    labels = tf.map_fn(make_labels, parts, back_prop=False, dtype=tf.float32)

    result = annot.copy()
    result["labels"] = to_channel_last(labels, batch=False)
    return result

matched_parts = [
	(1,4), (2,5), (10,14), (11,15), (12,16), (13,17), (22,23), (24,25),
    (29,30), (32,33), (34,36), (35,37), (38,39), (40,41), (42,44), (43,45),
    (46,48), (47,49), (50,51), (52,53), (54,55), (56,57), (58,59), (60,61),
    (62,63), (64,65), (66,67), (68,69), (70,71), (72,73), (74,75), (76,77),
    (78,80), (79,81), (83,85), (84,86), (87,89), (88,90), (91,93), (92,94),
    (95,96), (97,98), (99,100), (101,102)
]

# labels: channel first
def shuffleLR(labels):
    assert(len(labels.shape) == 3)
    def _shuffleLR(labels):
        res = [x for x in labels]
        for l, r in matched_parts:
            res[l-1], res[r-1] = res[r-1], res[l-1]
        return np.array(res, dtype=np.float32)

    return tf.py_func(_shuffleLR, [labels], tf.float32, stateful=False)
    

def augment(annot):
    input = annot["input"]
    labels = annot["labels"]

    # color variance
    input = tf.clip_by_value(
        input * tf.random_uniform([1, 1, 3], 0.8, 1.2),
        clip_value_min=0.0,
        clip_value_max=1.0)


    scale_factor = tf.to_float(opts.scale_factor)
    rot_factor = tf.to_float(opts.rot_factor)

    s = tf.clip_by_value(tf.random_normal([]) * scale_factor + 1.0,
                         clip_value_min=1.0-scale_factor,
                         clip_value_max=1.0+scale_factor)

    # rot_factor(degree) -> radian
    r = tf.clip_by_value(tf.random_normal([]) * rot_factor,
                         clip_value_min=-2.0 * rot_factor,
                         clip_value_max=2.0 * rot_factor) / 180.0 * math.pi

    # 60% 확률로 r == 0
    r = r * tf.to_float(tf.random_uniform([]) > 0.6)

    def resize_and_rotate(img, res, r, s):
        img = tf.image.resize_images(
            img, tf.to_int32(tf.to_float([res, res]) * s),
            method=tf.image.ResizeMethod.BICUBIC
        )
        # pad not to clip corners. cheap and fast math.sqrt(2)
        img = tf.image.resize_image_with_crop_or_pad(
            img, int(res * math.sqrt(2)), int(res * math.sqrt(2))
        )
        img = tf.contrib.image.rotate(img, r, interpolation='BILINEAR')
        img = tf.image.resize_image_with_crop_or_pad(
            img, res, res
        )
        return img

    input = resize_and_rotate(input, opts.input_res, r, s)
    labels = resize_and_rotate(labels, opts.output_res, r, s)

    flip = tf.random_uniform([], 0, 1) > 0.5
    input = tf.cond(
        flip,
        true_fn= lambda: input,
        false_fn= lambda: tf.image.flip_left_right(input)
    )

    def label_flip(x):
        x = to_channel_first(x, batch=False)
        x = shuffleLR(x)
        x = to_channel_last(x, batch=False)
        x = tf.image.flip_left_right(x)
        return x

    labels = tf.cond(
        flip,
        true_fn= lambda: labels,
        false_fn= lambda: label_flip(labels)
    )

    result = annot.copy()
    result["input"] = input
    result["labels"] = labels
    return result


def torch_batchnorm(x, training):
    return tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.9, training=training)


def conv2d(inputs, filters, kernel_size=(1, 1), strides=(1, 1), padding='SAME', name='conv'):
    with tf.name_scope(name):
        return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding)


def conv_block(x, num_out, training):
    with tf.name_scope("conv_block"):
        x = torch_batchnorm(x, training)
        x = tf.nn.relu(x)
        x = conv2d(x, num_out / 2, kernel_size=[1, 1])

        x = torch_batchnorm(x, training)
        x = tf.nn.relu(x)
        x = conv2d(x, num_out / 2, kernel_size=[3, 3])

        x = torch_batchnorm(x, training)
        x = tf.nn.relu(x)
        x = conv2d(x, num_out, kernel_size=[1, 1])
        return x


def skip_layer(x, num_out):
    num_in = x.get_shape()[3]
    if num_in == num_out:
        return x
    else:
        return conv2d(x, num_out, kernel_size=[1, 1])


def residual(x, num_out, training):
    with tf.name_scope("residual"):
        conv = conv_block(x, num_out, training)
        skip = skip_layer(x, num_out)
        return conv + skip


def hourglass(x, n, num_out, training):
    with tf.name_scope("hourglass{}".format(n)):
        upper = residual(x, 256, training)
        upper = residual(upper, 256, training)
        upper = residual(upper, num_out, training)

        lower = tf.layers.max_pooling2d(x, [2, 2], [2, 2])

        lower = residual(lower, 256, training)
        lower = residual(lower, 256, training)
        lower = residual(lower, 256, training)

        if n > 1:
            lower = hourglass(lower, n - 1, num_out, training)
        else:
            lower = residual(lower, num_out, training)

        lower = residual(lower, num_out, training)

        lower = tf.image.resize_nearest_neighbor(lower, tf.shape(lower)[1:3]*2)
        return upper + lower


def lin(x, num_out, training):
    with tf.name_scope("lin"):
        x = conv2d(x, num_out, kernel_size=[1, 1])
        x = tf.nn.relu(torch_batchnorm(x, training))
        return x


def stacked_hourglass(x, nparts, training):
    # preprocess
    with tf.name_scope("preprocess"):
        conv1 = conv2d(x, 64, [7, 7], strides=[2, 2], padding='SAME', name="256_to_128")  # 128
        conv1 = tf.nn.relu(torch_batchnorm(conv1, training))
        r1 = residual(conv1, 128, training)
        pool = tf.layers.max_pooling2d(r1, [2, 2], [2, 2])  # 64

        r4 = residual(pool, 128, training)
        r5 = residual(r4, 128, training)
        r6 = residual(r5, 256, training)

    hg1 = hourglass(r6, 4, 512, training)

    with tf.name_scope("out1"):
        # Linear layers to produce first set of predictions
        l1 = lin(hg1, 512, training)
        l2 = lin(l1, 256, training)

        # First predicted heatmaps
        out1 = conv2d(l2, nparts, [1, 1])
        out1_ = conv2d(out1, 256 + 128, [1, 1])

    # Concatenate with previous linear features
    cat1 = tf.concat([l2, pool], 3)  # concat channel
    cat1_ = conv2d(cat1, 256 + 128, [1, 1])

    int1 = out1_ + cat1_

    # Second hourglass
    hg2 = hourglass(int1, 4, 512, training)

    with tf.name_scope("out2"):
        # Linear layers to produce predictions again
        l3 = lin(hg2, 512, training)
        l4 = lin(l3, 512, training)

        # Output heatmaps
        out2 = conv2d(l4, nparts, [1, 1])

    return out1, out2


def get_preds(heatmap):
    reduce_x = tf.reduce_max(heatmap, axis=1)
    reduce_y = tf.reduce_max(heatmap, axis=2)
    idx_x = tf.expand_dims(tf.argmax(reduce_y, axis=1), axis=2)
    idx_y = tf.expand_dims(tf.argmax(reduce_x, axis=1), axis=2)
    return tf.concat([idx_x, idx_y], axis=2)


def calc_dists(preds, labels, normalize):
    stacked = tf.stack([preds, labels], axis=2)

    def map_channel(x):
        pred = x[0]
        label = x[1]

        dist = tf.reduce_sum(tf.norm(tf.to_float(pred - label)) / normalize)
        return tf.cond(
            tf.logical_and(tf.greater(label[0], 0), tf.greater(label[1], 0)),
            true_fn=lambda: dist,
            false_fn=lambda: tf.constant(-1, dtype=tf.float32))

    def map_batch(x):
        return tf.map_fn(map_channel, x, back_prop=False, dtype=tf.float32)

    # (batch, channel, [pred, label], [x, y])
    # -> (batch, channel)
    dists = tf.map_fn(map_batch, stacked, back_prop=False, dtype=tf.float32)
    return tf.transpose(dists)


def dist_accuracy(dists):
    thr = 0.5
    valid = tf.greater_equal(dists, 0)
    thres = tf.less(tf.to_float(dists), tf.to_float(thr))
    total_count = tf.reduce_sum(tf.to_float(valid))
    thres_count = tf.reduce_sum(tf.to_float(tf.logical_and(thres, valid)))
    return tf.cond(
        tf.greater(total_count, 0),
        true_fn=lambda: thres_count / total_count,
        false_fn=lambda: tf.constant(-1, dtype=tf.float32))


def heatmapAccuracy(output, label):
    preds = get_preds(output)
    gt = get_preds(label)  # ground truth
    dists = calc_dists(preds, gt, normalize=tf.to_float(opts.output_res / 10.0)) # 10? why 10?
    acc = tf.map_fn(dist_accuracy, dists, back_prop=False)
    valid_sum = tf.reduce_sum(tf.maximum(acc, 0.0))
    valid_count = tf.reduce_sum(tf.to_float(tf.not_equal(acc, -1.0)))
    return valid_sum / valid_count


def summary_label(label):
    def sum(channels):
        return tf.reduce_sum(channels, axis=0, keep_dims=False)

    channel_first = to_channel_first(label) # (batch, H, W, chan) -> (batch, chan, H, W)
    summary = tf.map_fn(sum, channel_first, back_prop=False) # (batch, H, W)
    return tf.expand_dims(summary, axis=3)


def filtered_summary(groundtruth, heatmap):
    def filter_hm(accum, stacked):
        gt = stacked[0]
        hm = stacked[1]
        max = tf.reduce_max(hm)
        min = tf.reduce_min(hm)
        hm = tf.pow((hm-min) / (max-min), 2) / 2
        hm_out = tf.expand_dims(hm, axis=2) # add channel

        return tf.cond(
            tf.not_equal(tf.reduce_sum(gt), 0),
            true_fn=lambda: accum + hm_out,
            false_fn=lambda: accum,
        )

    def filter_pred(accum, stacked):
        gt = stacked[0]
        hm = stacked[1]
        max = tf.reduce_max(hm)

        zeros =  tf.zeros(gt.get_shape())
        hm_pred = tf.to_float(tf.equal(hm, max))
        hm_pred = tf.stack([hm_pred, zeros, zeros], axis=2)
        return tf.cond(
            tf.not_equal(tf.reduce_sum(gt), 0),
            true_fn=lambda: accum + hm_pred,
            false_fn=lambda: accum,
        )

    def filter_sum(channels):
        gt_shape = channels[0][0].get_shape()
        hm = tf.foldl(filter_hm, channels,
                        initializer=tf.zeros([gt_shape[0], gt_shape[1], 3]),
                        back_prop=False)
        hm = tf.cond(
            (tf.reduce_max(hm) - tf.reduce_min(hm)) > 0.01,
            true_fn = lambda: (hm - tf.reduce_min(hm))/(tf.reduce_max(hm) - tf.reduce_min(hm)),
            false_fn = lambda: tf.zeros(tf.shape(hm))
        )

        pred = tf.foldl(filter_pred, channels,
                      initializer=tf.zeros([gt_shape[0], gt_shape[1], 3]),
                      back_prop=False)

        return tf.clip_by_value(hm + pred, clip_value_min=0, clip_value_max=1)

    # (batch, H, W, chan) -> (batch, chan, H, W)
    channel_first_groundtruth = to_channel_first(groundtruth)
    channel_first_heatmap = to_channel_first(heatmap)

    stacked = tf.stack([channel_first_groundtruth, channel_first_heatmap], axis=2) # (batch, chan, 2, H, W)
    summary = tf.map_fn(filter_sum, stacked, back_prop=False) # (batch, H, W, channels)
    return summary

import numpy as np
import math

def heatmap_thumbs(heatmaps):
    def plot(heatmaps):
        count = int(math.ceil(math.sqrt(heatmaps.shape[0])))
        h = heatmaps.shape[1]
        w = heatmaps.shape[2]

        max = np.max(heatmaps)
        min = np.min(heatmaps)

        canvas = np.ones([int(count * h * 1.1), int(count * w * 1.1)], dtype=np.float32)
        for i in range(heatmaps.shape[0]):
            top = int(int(i / count) * h * 1.1)
            left = int((i % count) * w * 1.1)

            if math.fabs(max-min) < 0.01:
                hm_out = np.zeros([w, h], dtype=np.float32)
            else:
                hm_out = (heatmaps[i]-min)/(max-min)
            canvas[top:top+h, left:left+w] = hm_out
        return canvas

    heatmaps = to_channel_first(heatmaps)

    def py_plot(heatmaps):
        return tf.clip_by_value(tf.expand_dims(tf.py_func(plot, [heatmaps], tf.float32, False), axis=2), 0, 1)

    return tf.map_fn(py_plot, heatmaps, back_prop=False, dtype=tf.float32)


def model_fn(features, labels, mode):
    input = features["input"]

    out1, out2 = stacked_hourglass(features["input"], features["ref"]["nparts"], mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={
            "index": features["index"],
            "heatmaps": out2
        })

    loss = tf.losses.mean_squared_error([labels, labels], [out1, out2])
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=[out1, out2], labels=[labels, labels]))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print(update_ops)
    global_step= tf.train.get_global_step()
    optimizer = tf.train.RMSPropOptimizer(2.5e-4)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    tf.summary.scalar("accuracy", heatmapAccuracy(out2, labels))
    tf.summary.scalar("loss", loss)

    tf.summary.image("image", input)
    tf.summary.image("labels", summary_label(labels))
    tf.summary.image("labels-thumb", heatmap_thumbs(labels))
    tf.summary.histogram("labels-hist", labels)

    tf.summary.image("out1", summary_label(out1))
    out2_summary = summary_label(out2)
    tf.summary.image("out2", out2_summary)
    tf.summary.histogram("out2-summary-hist", out2_summary)
    tf.summary.histogram("out2-hist", out2)

    tf.summary.image("out1-filtered", filtered_summary(labels, out1))
    tf.summary.image("out2-filtered", filtered_summary(labels, out2))

    tf.summary.image("out1-thumb", heatmap_thumbs(out1))
    tf.summary.image("out2-thumb", heatmap_thumbs(out2))

    # summary에 image, sum of labels, sum of out2 출력

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            "heatmaps": out2,
        },
        loss=loss,
        train_op=train_op,
        training_hooks=[
            tf.train.SummarySaverHook(
                save_secs=10,
                output_dir=opts.log_dir,
                summary_op=tf.summary.merge_all()
            ),
        ],
        eval_metric_ops={
            "accuracy": (heatmapAccuracy(out2, labels), tf.no_op())
        }
    )


cheat_part_class = {
    'aeroplane': {0, 1, 2, 3, 4, 5, 6, 7},
    'bicycle': {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
    'boat': {19, 20, 21, 22, 23, 24, 25},
    'bottle': {26, 27, 28, 29, 30, 31, 32},
    'bus': {33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44},
    'car': {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56},
    'chair': {57, 58, 59, 60, 61, 62, 63, 64, 65, 66},
    'sofa': {67, 68, 69, 70, 71, 72, 73, 74, 75, 76},
    'train': {77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93},
    'tvmonitor': {94, 95, 96, 97, 98, 99, 100, 101}
}


def only_car(annot):
    def _only_car(idx, part):
        valid_part_idxs = [i for i in range(len(part)) if part[i][0] != -1]
        idx_in_car = [idx in cheat_part_class["car"] for idx in valid_part_idxs]

        if len(idx_in_car) > 0 and all(idx_in_car) :
            return True
        else:
            return False

    return tf.py_func(_only_car, [annot["index"], tf.to_int32(annot["part"])], stateful=False, Tout=tf.bool)


def train_input_fn():
    train_annots = load_annots("train")

    BATCH = 4
    STEPS = 4000

    dataset = train_annots["dataset"] \
        .take(BATCH * STEPS) \
        .repeat() \
        .filter(only_car) \
        .shuffle(BATCH * STEPS) \
        .map(make_feature_and_labels, num_parallel_calls=8) \
        .map(augment, num_parallel_calls=8) \
        .prefetch(16) \
        .batch(4)

    data = dataset.make_one_shot_iterator().get_next()
    # batch channel width height
    # -> batch width height channel

    features = {
        "input": data["input"],
        "ref": train_annots["ref"]
    }
    return features, data["labels"]


def train_main():
    # https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    model = tf.estimator.Estimator(model_fn, model_dir=opts.model_dir,
                                   config=tf.estimator.RunConfig(session_config=config))

    model.train(train_input_fn, steps=4000 * 100)


def pred_input_fn():
    train_annots = load_annots("valid")

    dataset = train_annots["dataset"] \
        .map(make_input_crop, num_parallel_calls=8) \
        .batch(16)

    data = dataset.make_one_shot_iterator().get_next()

    features = {
        "index": data["index"],
        "input": data["input"],
        "ref": train_annots["ref"]
    }
    return features


def pred_main():
    model = tf.estimator.Estimator(model_fn, model_dir=opts.model_dir,
                                   config=tf.estimator.RunConfig(session_config=config))

    import pathlib
    pathlib.Path("exp/pascal3d/").mkdir(parents=True, exist_ok=True)

    for output in model.predict(pred_input_fn):
        with h5py.File(path.join("exp/pascal3d/",
                                 "valid_{}.h5".format(output["index"])),
                       'w') as f:
            f["heatmaps"] = np.transpose(output["heatmaps"], (2, 0, 1))


def dummy_main():
    train_annots = load_annots("valid")

    dataset = train_annots["dataset"] \
        .map(make_feature_and_labels, num_parallel_calls=8) \
        .prefetch(16) \
        .batch(16)

    data = dataset.make_one_shot_iterator().get_next()

    import pathlib
    pathlib.Path("exp/pascal3d/").mkdir(parents=True, exist_ok=True)

    with tf.Session() as sess:
        while True:
            output = sess.run({
             "index": data["index"],
             "heatmaps": data["labels"]
            })

            for i in range(len(output["index"])):
                filename = "valid_{}.h5".format(output["index"][i])
                filepath = path.join("exp/pascal3d/", filename)
                with h5py.File(filepath, 'w') as f:
                    f["heatmaps"] = np.transpose(output["heatmaps"][i], (2, 0, 1))


if __name__ == "__main__":
    if opts.mode == "train":
        train_main()
    elif opts.mode == "pred":
        pred_main()
    elif opts.mode == "dummy":
        dummy_main()
    else:
        raise ValueError("Invalid mode " + opts.mode)

