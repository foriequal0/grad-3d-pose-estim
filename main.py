import os
from os import path
import h5py
import tensorflow as tf
from stacked_hourglass.opts import opts

tf.logging.set_verbosity(tf.logging.INFO)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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


def decode_and_crop(annot):
    filename = annot["image"]
    image_path = tf.string_join([opts.data_dir, "pascal3d", "images", filename], separator="/")
    parts = tf.to_float(annot["part"])
    center = tf.to_float(annot["center"])
    scale = tf.to_float(annot["scale"]) / 1.28  # 1.28 is 256/200, original paper impl.'s flaw

    image_string = tf.read_file(image_path)
    image_decoded = tf.cast(tf.image.decode_image(image_string, channels=3), tf.float32) / 255.0
    image_decoded.set_shape([None, None, 3])  # decode_image doesn't set channels correctly

    image_cropped = crop(image_decoded, center, scale)

    def make_labels(part):
        return tf.cond(
            tf.greater(part[1], 0),
            true_fn=lambda: draw_gaussian(part, center, scale, 1.75),
            false_fn=lambda: tf.zeros([64, 64], tf.float32))

    labels = tf.map_fn(make_labels, parts, back_prop=False, dtype=tf.float32)

    result = annot.copy()
    result["input"] = image_cropped
    result["labels"] = labels
    return result


def conv_block(x, num_out):
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, num_out / 2, kernel_size=[1, 1])

    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, num_out / 2, kernel_size=[3, 3], padding='SAME')

    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, num_out, kernel_size=[1, 1])
    return x


def skip_layer(x, num_out):
    num_in = x.get_shape()[3]
    if num_in == num_out:
        return tf.identity(x)
    else:
        return tf.layers.conv2d(x, num_out, kernel_size=[1, 1])


def residual(x, num_out):
    conv = conv_block(x, num_out)
    skip = skip_layer(x, num_out)
    return conv + skip


def hourglass(x, n, num_out):
    upper = residual(x, 256)
    upper = residual(upper, 256)
    upper = residual(upper, num_out)

    lower = tf.layers.max_pooling2d(x, [2, 2], [2, 2])
    lower = residual(lower, 256)
    lower = residual(lower, 256)
    lower = residual(lower, 256)

    if n > 1:
        lower = hourglass(lower, n - 1, num_out)
    else:
        lower = residual(lower, num_out)

    lower = residual(lower, num_out)
    lower = tf.keras.layers.UpSampling2D([2, 2]).apply(lower)

    return lower + upper


def lin(x, num_out):
    x = tf.layers.conv2d(x, num_out, kernel_size=[1, 1])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    return x


def stacked_hourglass(x, nparts):
    # preprocess
    conv1 = tf.layers.conv2d(x, 64, kernel_size=[7, 7], strides=[2, 2], padding='SAME')  # 128

    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)

    r1 = residual(conv1, 128)

    pool = tf.layers.max_pooling2d(r1, [2, 2], [2, 2])  # 64

    r4 = residual(pool, 128)
    r5 = residual(r4, 128)
    r6 = residual(r5, 256)

    hg1 = hourglass(r6, 4, 512)

    # Linear layers to produce first set of predictions
    l1 = lin(hg1, 512)
    l2 = lin(l1, 256)

    # First predicted heatmaps
    out1 = tf.layers.conv2d(l2, nparts, kernel_size=[1, 1])
    out1_ = tf.layers.conv2d(l2, 256 + 128, kernel_size=[1, 1])

    # Concatenate with previous linear features
    cat1 = tf.concat([l2, pool], 3)  # concat channel
    cat1_ = tf.layers.conv2d(cat1, 256 + 128, kernel_size=[1, 1])

    int1 = out1_ + cat1_

    # Second hourglass
    hg2 = hourglass(int1, 4, 512)
    # Linear layers to produce predictions again
    l3 = lin(hg2, 512)
    l4 = lin(l3, 512)

    # Output heatmaps
    out2 = tf.layers.conv2d(l4, nparts, kernel_size=[1, 1])

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

        dist = tf.reduce_sum(tf.to_float(tf.pow(pred - label, 2)) / normalize)
        return tf.cond(
            tf.logical_and(label[0] > 0, label[1] > 0),
            true_fn=lambda: dist,
            false_fn=lambda: tf.constant(-1, dtype=tf.float32))

    def map_batch(x):
        return tf.map_fn(map_channel, x, back_prop=False, dtype=tf.float32)

    return tf.transpose(tf.map_fn(map_batch, stacked, back_prop=False, dtype=tf.float32))


def dist_accuracy(dists):
    thr = 0.5
    valid = tf.not_equal(dists, -1)
    thres = tf.less(tf.to_float(dists), tf.to_float(thr))
    total_count = tf.reduce_sum(tf.to_float(valid))
    thres_count = tf.reduce_sum(tf.to_float(tf.logical_and(thres, valid)))
    return tf.cond(
        total_count > 0,
        true_fn=lambda: total_count / thres_count,
        false_fn=lambda: tf.constant(-1, dtype=tf.float32))


def heatmapAccuracy(output, label):
    preds = get_preds(output)
    gt = get_preds(label) # ground truth
    dists = calc_dists(preds, gt, normalize=tf.to_float(opts.output_res / 10.0))
    acc = tf.map_fn(dist_accuracy, dists, back_prop=False)
    valid_sum = tf.reduce_sum(tf.maximum(acc, 0.0))
    valid_count = tf.reduce_sum(tf.to_float(tf.not_equal(acc, -1.0)))
    return valid_sum / valid_count



def model_fn(features, labels, mode):
    out1, out2 = stacked_hourglass(features["input"], features["ref"]["nparts"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=out2)

    loss = tf.losses.mean_squared_error([labels, labels], [out1, out2])
    train_op = tf.train.RMSPropOptimizer(2.5e-4) \
        .minimize(loss, global_step=tf.train.get_global_step())

    tf.summary.scalar("accuracy", heatmapAccuracy(out2, labels))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=out2,
        loss=loss,
        train_op=train_op,
        training_hooks=[
            tf.train.SummarySaverHook(
                save_steps=1,
                output_dir="./model",
                summary_op=tf.summary.merge_all()
            ),
        ],
        eval_metric_ops={
            "accuracy": (heatmapAccuracy(out2, labels), tf.no_op())
        }
    )


# https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html

model = tf.estimator.Estimator(model_fn, model_dir=opts.model_dir)

def train_input_fn():
    # TODO: move to somewhere
    train_annots = load_annots("train")

    dataset = train_annots["dataset"] \
        .map(decode_and_crop, num_parallel_calls=8) \
        .repeat() \
        .prefetch(32) \
        .batch(4)

    data = dataset.make_one_shot_iterator().get_next()
    # batch channel width height
    # -> batch width height channel
    multichannel = tf.transpose(data["labels"], perm=[0, 2, 3, 1])
    features = {
        "input": data["input"],
        "ref": train_annots["ref"]
    }
    return features, multichannel


for epoch in range(100):
    print("epoch ", epoch)
    model.train(train_input_fn, steps=4000)
