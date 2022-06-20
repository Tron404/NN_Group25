import numpy as np
import tensorflow as tf
from keras import backend

def custom_huber(y_true, y_pred):
    delta = tf.cast(1.0, dtype=backend.floatx())
    y_true_tf = tf.cast(y_true, dtype=backend.floatx())
    y_pred_tf = tf.cast(y_pred, dtype=backend.floatx())
    diff = tf.subtract(y_pred_tf, y_true_tf)
    abs_diff = tf.abs(diff)

    half = tf.convert_to_tensor(0.5, dtype=abs_diff.dtype)
    one_half = tf.convert_to_tensor(1.5, dtype=abs_diff.dtype)

    hbl = tf.where(abs_diff <= delta, half * tf.square(diff), delta * abs_diff - half * tf.square(delta))

    pressure = tf.where(tf.divide(y_true_tf, y_pred_tf) <= 1.0, 0.0, one_half * tf.divide(y_true_tf, y_pred_tf))
    # hbl_pressure = tf.divide(pressure, hbl)

    print("diff ", diff)
    print("hbl ", hbl)
    print("pressure ", pressure)

    return backend.mean(hbl+pressure, axis=-1)

y_true = [1.0, 0.5, 0.5, 0.1, 1.0, 0.5]
y_pred = [1.0, 1.0, 0.9, 0.8, 0.1, 0.4]

custom_huber(y_true, y_pred)