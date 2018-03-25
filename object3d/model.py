import tensorflow as tf
import math

def torch_batchnorm(x, training):
    return tf.layers.batch_normalization(
        x, epsilon=1e-5, momentum=0.9, training=training,
        gamma_initializer=tf.random_uniform_initializer(0, 1),
        beta_initializer=tf.zeros_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer()
    )


def conv2d(inputs, filters, kernel_size=(1, 1), strides=(1, 1), padding='SAME', name='conv'):
    with tf.name_scope(name):
        stdv = 1/math.sqrt(kernel_size[0]*kernel_size[1]*filters)

        return tf.layers.conv2d(
            inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding,
            kernel_initializer=tf.random_uniform_initializer(-stdv, stdv),
            bias_initializer=tf.random_uniform_initializer(-stdv, stdv),
        )


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
