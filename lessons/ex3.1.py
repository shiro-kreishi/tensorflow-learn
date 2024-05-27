# for exec on cpu
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


if __name__ == '__main__':

    x = tf.Variable(0.0)

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(x)
        y = 2 * x
        f = y * y

    df = tape.gradient(f, y)
    df_dx = tape.gradient(f, x)

    del tape
    print(df, df_dx, sep='\n')