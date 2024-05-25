# for exec on cpu
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    a = tf.zeros((3, 3))
    print(a)
    b = tf.ones((5, 3))
    print(b)
    c = tf.ones_like(a)
    print(c)
    d = tf.eye(3, 2)
    print(d)
    f = tf.fill((3, 3), -1)
    print(f)
    g = tf.range(1, 10, 0.2)
    print(g)

    r_normal = tf.random.normal((5, 3), 0, 0.1)
    print(r_normal)
    r_uniform = tf.random.uniform((3, 3), 0, 0.1)
    print(r_uniform)
    tf.random.set_seed(0)
    r_truncated = tf.random.truncated_normal((5, 5), 0, 0.1)
    print(r_truncated)
    tf.random.set_seed(1)
    r_truncated = tf.random.truncated_normal((5, 5), 0, 0.1)
    print(r_truncated)

    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    print(a, b, sep='\n')
    sum = tf.add(a, b)      # a+b
    print(sum)
    sub = tf.subtract(a, b)     # a-b
    print(sub)
    mul = tf.multiply(a, b)     # a*b
    print(mul)
    div = tf.divide(a, b)       # a/b
    print(div)
    print(a**b)
    t_dot1 = tf.tensordot(a, b, axes=0)
    t_dot2 = tf.tensordot(a, b, axes=1)
    print(t_dot1, t_dot2, sep='\n')

    a2 = tf.constant(tf.range(1, 10), shape=[3, 3])
    b2 = tf.constant(tf.range(5, 14), shape=[3, 3])
    print(a2, b2, tf.matmul(a2, b2), sep='\n') # произведение двух матриц или a2 @ b2

    m = tf.tensordot(a, b, axes=0)
    print(m,
          tf.reduce_sum(m), tf.reduce_mean(m),
          tf.reduce_max(m), tf.reduce_min(m),
          tf.reduce_max(m),
          sep='\n')
    print(
        tf.sqrt(tf.cast(a, dtype=tf.float32)),
        tf.square(tf.cast(a, dtype=tf.float32)),
        tf.sign(tf.cast(a, dtype=tf.float32)),
        tf.cos(tf.cast(a, dtype=tf.float32)),

        sep='\n'
    )