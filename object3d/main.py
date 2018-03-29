# weight initializer 맞추기
# loss 생성시 해당 채널만 mean 하기.
# loss를 sigmoid_cross_entrophy

import numpy as np
import h5py
import os.path as path
import tensorflow as tf

from .opts import opts
from . import model
from . import summary
from . import loader

def model_fn(features, labels, mode):
    input = features["input"]

    out1, out2 = model.stacked_hourglass(features["input"], features["ref"]["nparts"], mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={
            "index": features["index"],
            "heatmaps": out2
        })

    loss = tf.losses.mean_squared_error([labels, labels], [out1, out2])
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=[out1, out2], labels=[labels, labels]))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step= tf.train.get_global_step()

    from . import torch_rmsprop
    if opts.optimizer == 'torch':
        optimizer = torch_rmsprop.RMSPropOptimizer(2.5e-4, use_locking=True)
    elif opts.optimizer == 'tf':
        optimizer = tf.train.RMSPropOptimizer(2.5e-4, epsilon=1e-16, decay=0.99, use_locking=True)
    else:
        raise ValueError("optimizer {} is invalid".format(opts.optimizer))
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    tf.summary.scalar("accuracy", summary.heatmapAccuracy(out2, labels))
    tf.summary.scalar("loss", loss)

    tf.summary.image("image", input)
    tf.summary.histogram("image", input)
    tf.summary.image("labels", summary.summary_label(labels))
    tf.summary.image("labels-thumb", summary.heatmap_thumbs(labels))
    tf.summary.histogram("labels-hist", labels)

    tf.summary.image("out1", summary.summary_label(out1))
    out2_summary = summary.summary_label(out2)
    tf.summary.image("out2", out2_summary)
    tf.summary.histogram("out2-summary-hist", out2_summary)
    tf.summary.histogram("out2-hist", out2)

    tf.summary.image("out1-filtered", summary.filtered_summary(labels, out1))
    tf.summary.image("out2-filtered", summary.filtered_summary(labels, out2))

    tf.summary.image("out1-thumb", summary.heatmap_thumbs(out1))
    tf.summary.image("out2-thumb", summary.heatmap_thumbs(out2))

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
            "accuracy": (summary.heatmapAccuracy(out2, labels), tf.no_op())
        }
    )


BATCH = opts.train_batch
STEPS = opts.train_iters
INPUTS = STEPS * 4

def train_input_fn():
    train_annots = loader.load_annots("train")

    dataset = train_annots["dataset"] \
        .take(INPUTS) \
        .repeat() \
        .shuffle(INPUTS) \
        .map(loader.make_input_and_labels, num_parallel_calls=8) \
        .map(loader.augment, num_parallel_calls=8) \
        .prefetch(BATCH*2) \
        .batch(BATCH)

    data = dataset.make_one_shot_iterator().get_next()
    # batch channel width height
    # -> batch width height channel

    features = {
        "input": data["input"],
        "ref": train_annots["ref"]
    }
    return features, data["labels"]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def train_main():
    # https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    model = tf.estimator.Estimator(model_fn, model_dir=opts.model_dir,
                                   config=tf.estimator.RunConfig(session_config=config))

    model.train(train_input_fn, steps=STEPS * 4 / BATCH * 100)


def pred_input_fn():
    train_annots = loader.load_annots("valid")

    dataset = train_annots["dataset"] \
        .map(loader.make_input, num_parallel_calls=8) \
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
    train_annots = loader.load_annots("valid")

    dataset = train_annots["dataset"] \
        .map(loader.make_input_and_labels, num_parallel_calls=8) \
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
