# for exec on cpu
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    a = tf.constant(1, shape=(1, 1))
    print(a)
    b = tf.constant(2, shape=(2, 2))
    print(b)
    c = tf.constant([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=tf.float16)
    print(c)
    # cast tensor to tensor
    a2 = tf.cast(a, dtype=tf.float16)
    print(a2)
    # create mutable tensor
    v2 = tf.Variable([1, 2, 3], dtype=tf.float16)
    v1 = tf.Variable(1, dtype=tf.float16)
    v3 = tf.Variable([1, 2, 3, 4])
    print(v1, v2, v3, sep="\n\n")
    print('assign: ', sep='\n')
    v1.assign(0)
    v2.assign([4, 5, 6])
    print(v1, v2, sep="\n", end='\n\n')

    val_0 = v3[0]       # first element
    val_12 = v3[1:3]    # element 2-3
    val_0.assign(10)
    print(v3, val_0, val_12, sep="\n")

    v4 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v4_index = v4[:2, -1]     # start:stop:step
    print(v4, v4_index, sep="\n")

    a = tf.constant(range(30))
    print(a)
    b = tf.reshape(a, [6, -1])
    # b = tf.reshape(a, [6, 5])
    print(b.numpy())
    b_T = tf.transpose(b, perm=[1, 0])
    print(b_T.numpy())
