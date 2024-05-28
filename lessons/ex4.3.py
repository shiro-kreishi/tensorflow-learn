import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

TOTAL_POINTS = 1000

if __name__ == '__main__':
    x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
    noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)
    k_true = 0.7
    b_true = 2.0

    y = x * k_true + b_true + noise
    # plt.scatter(x, y, s=2)
    # plt.show()

    k, b = tf.Variable(0.0), tf.Variable(0.0)

    EPOCHS = 50
    learning_rate = 0.02
    BATCH_SIZE = 100
    num_steps = TOTAL_POINTS // BATCH_SIZE

    # opt = tf.optimizers.SGD(learning_rate=learning_rate, nesterov=True, momentum=0.5)
    # opt = tf.optimizers.Adagrad(learning_rate=0.1)
    # opt = tf.optimizers.Adadelta(learning_rate=2.0)
    # opt = tf.optimizers.RMSprop(learning_rate=0.01)
    opt = tf.optimizers.Adam(learning_rate=0.1)

    for epoch in range(EPOCHS):
        for n_batch in range(num_steps):
            y_batch = y[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE]
            x_batch = x[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE]

            with tf.GradientTape() as tape:
                f = k * x_batch + b
                loss = tf.reduce_mean(tf.square(y_batch - f))

            dk, db = tape.gradient(loss, [k, b])
            opt.apply_gradients(zip([dk, db], [k, b]))
            # k.assign_sub(learning_rate * dk)
            # b.assign_sub(learning_rate * db)

    print(k, b, sep='\n')
    y_pr = k * x + b
    plt.scatter(x, y, s=2)
    plt.scatter(x, y_pr, c='r', s=2)
    plt.show()