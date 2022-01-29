import tensorflow as tf 


def db_scale(input, top_db, name=None):
    power = tf.math.square(input)
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec

def trim(input, axis, epsilon, name=None):
    """
    Trim the noise from beginning and end of the audio.
    Args:
      input: An audio Tensor.
      axis: The axis to trim.
      epsilon: The max value to be considered as noise.
      name: A name for the operation (optional).
    Returns:
      A tensor of start and stop with shape `[..., 2, ...]`.
    """
    shape = tf.shape(input, out_type=tf.int64)
    length = shape[axis]

    nonzero = tf.math.greater(input, epsilon)
    check = tf.reduce_any(nonzero, axis=axis)

    forward = tf.cast(nonzero, tf.int8)
    reverse = tf.reverse(forward, [axis])

    start = tf.where(check, tf.argmax(forward, axis=axis), length)
    stop = tf.where(check, tf.argmax(reverse, axis=axis), tf.constant(0, tf.int64))
    stop = length - stop

    return tf.stack([start, stop], axis=axis)

