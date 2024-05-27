import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def scalar_input():
    x = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        y = [2.0, 3.0] * x ** 2
    df = tape.gradient(y, x)
    print(df)


def vector_input():
    x = tf.Variable([1.0, 2.0])
    with tf.GradientTape() as tape:
        y = tf.reduce_sum([3.0, 4.0]) * x ** 2
    df = tape.gradient(y, x)
    print(df)


def vector_input_in_point():
    x = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        if x < 2.0:
            y = tf.reduce_sum([2.0, 3.0]) * x ** 2
        else:
            y = x ** 2
    df = tape.gradient(y, x)
    print(df)


if __name__ == '__main__':
    print('tensorflow version: ', tf.__version__)
    scalar_input()
    vector_input()
    vector_input_in_point()
