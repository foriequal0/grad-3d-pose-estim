import tensorflow as tf


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


def py_func(Tout=None, stateful=True, name=None):
    def decorator(func):
        def wrapper(*args, Tout=Tout, stateful=stateful, name=name):
            return tf.py_func(func, list(args), Tout=Tout, stateful=stateful, name=name)
        return wrapper
    return decorator