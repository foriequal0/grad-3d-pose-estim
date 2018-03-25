import tensorflow as tf
import numpy as np
import math

from . import util
from .opts import opts


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

    channel_first = util.to_channel_first(label) # (batch, H, W, chan) -> (batch, chan, H, W)
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
    channel_first_groundtruth = util.to_channel_first(groundtruth)
    channel_first_heatmap = util.to_channel_first(heatmap)

    stacked = tf.stack([channel_first_groundtruth, channel_first_heatmap], axis=2) # (batch, chan, 2, H, W)
    summary = tf.map_fn(filter_sum, stacked, back_prop=False) # (batch, H, W, channels)
    return summary

def heatmap_thumbs(heatmaps):
    @util.py_func(Tout=tf.float32, stateful=False)
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

    heatmaps = util.to_channel_first(heatmaps)

    def plot_per_batch(heatmaps):
        return tf.clip_by_value(tf.expand_dims(plot(heatmaps), axis=2), 0, 1)

    return tf.map_fn(plot_per_batch, heatmaps, back_prop=False, dtype=tf.float32)
