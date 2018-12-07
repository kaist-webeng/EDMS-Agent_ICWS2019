import tensorflow as tf


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def variable_summaries(var, varname):
    with tf.name_scope(varname):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)
