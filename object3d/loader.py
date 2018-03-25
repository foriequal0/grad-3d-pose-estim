import tensorflow as tf
import numpy as np
import math
import os.path as path
import h5py

from . import util
from .opts import opts


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


def _crop(image, center, scale):
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


def _draw_gaussian(pos, center, scale, sigma):
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


def make_input(annot):
    filename = annot["image"]
    image_path = tf.string_join([opts.data_dir, "pascal3d", "images", filename], separator="/")
    center = tf.to_float(annot["center"])
    scale = tf.to_float(annot["scale"]) / 1.28  # 1.28 is 256/200, original paper impl.'s flaw

    image_string = tf.read_file(image_path)
    image_decoded = tf.cast(tf.image.decode_image(image_string, channels=3), tf.float32) / 255.0
    image_decoded.set_shape([None, None, 3])  # decode_image doesn't set channels correctly

    image_cropped = _crop(image_decoded, center, scale)

    result = annot.copy()
    result["input"] = image_cropped
    return result


def make_input_and_labels(annot):
    annot = make_input(annot)

    parts = tf.to_float(annot["part"])
    center = tf.to_float(annot["center"])
    scale = tf.to_float(annot["scale"]) / 1.28  # 1.28 is 256/200, original paper impl.'s flaw

    def make_labels(part):
        return tf.cond(
            tf.greater(part[1], 0),
            true_fn=lambda: _draw_gaussian(part, center, scale, 1.75),
            false_fn=lambda: tf.zeros([64, 64], tf.float32))

    labels = tf.map_fn(make_labels, parts, back_prop=False, dtype=tf.float32)

    result = annot.copy()
    result["labels"] = util.to_channel_last(labels, batch=False)
    return result


_parts_pair = [
    (1, 4), (2, 5), (10, 14), (11, 15), (12, 16), (13, 17), (22, 23), (24, 25),
    (29, 30), (32, 33), (34, 36), (35, 37), (38, 39), (40, 41), (42, 44), (43, 45),
    (46, 48), (47, 49), (50, 51), (52, 53), (54, 55), (56, 57), (58, 59), (60, 61),
    (62, 63), (64, 65), (66, 67), (68, 69), (70, 71), (72, 73), (74, 75), (76, 77),
    (78, 80), (79, 81), (83, 85), (84, 86), (87, 89), (88, 90), (91, 93), (92, 94),
    (95, 96), (97, 98), (99, 100), (101, 102)
]


# labels: channel first

def _shuffle_left_right(labels):
    assert (len(labels.shape) == 3)

    @util.py_func(Tout=tf.float32, stateful=False)
    def _shuffle(labels):
        res = [x for x in labels]
        for l, r in _parts_pair:
            res[l - 1], res[r - 1] = res[r - 1], res[l - 1]
        return np.array(res, dtype=np.float32)

    return _shuffle(labels)


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
                         clip_value_min=1.0 - scale_factor,
                         clip_value_max=1.0 + scale_factor)

    # rot_factor(degree) -> radian
    r = tf.clip_by_value(tf.random_normal([]) * rot_factor,
                         clip_value_min=-2.0 * rot_factor,
                         clip_value_max=2.0 * rot_factor) / 180.0 * math.pi

    # 60% 확률로 r == 0
    r = r * tf.to_float(tf.random_uniform([]) > 0.6)

    def resize_and_rotate(img, res, r, s):
        img = tf.image.resize_images(
            img, tf.to_int32(tf.to_float([res, res]) * s)
        )
        # pad not to clip corners. cheap and fast math.sqrt(2)
        img = tf.image.resize_image_with_crop_or_pad(
            img, int(res * math.sqrt(2)), int(res * math.sqrt(2))
        )
        img = tf.contrib.image.rotate(img, r)
        img = tf.image.resize_image_with_crop_or_pad(
            img, res, res
        )
        return img

    input = resize_and_rotate(input, opts.input_res, r, s)
    labels = resize_and_rotate(labels, opts.output_res, r, s)

    flip = tf.random_uniform([], 0, 1) > 0.5
    input = tf.cond(
        flip,
        true_fn=lambda: input,
        false_fn=lambda: tf.image.flip_left_right(input)
    )

    def label_flip(x):
        x = util.to_channel_first(x, batch=False)
        x = _shuffle_left_right(x)
        x = util.to_channel_last(x, batch=False)
        x = tf.image.flip_left_right(x)
        return x

    labels = tf.cond(
        flip,
        true_fn=lambda: labels,
        false_fn=lambda: label_flip(labels)
    )

    result = annot.copy()
    result["input"] = input
    result["labels"] = labels
    return result


_cheat_parts_of_class = {
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
    @util.py_func(Tout=tf.bool, stateful=False)
    def _only_car(part):
        valid_part_idxs = [i for i in range(len(part)) if part[i][0] != -1]
        idx_in_car = [idx in _cheat_parts_of_class["car"] for idx in valid_part_idxs]

        if len(idx_in_car) > 0 and all(idx_in_car) :
            return True
        else:
            return False

    return _only_car(tf.to_int32(annot["part"]))

