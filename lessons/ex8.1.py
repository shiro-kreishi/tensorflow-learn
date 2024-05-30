import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class NeuralNetwork(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = DenseLayer(128)
        self.layer2 = DenseLayer(10)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        x = tf.nn.softmax(x)
        return x

if __name__ == '__main__':
    model = NeuralNetwork()
    # model.compile(
    #     loss=tf.losses.categorical_crossentropy,
    #     optimizer=tf.optimizers.Adam(learning_rate=0.001),
    #     metrics=['accuracy']
    # )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy']
    )
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
    x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

    y_train = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    model.fit(x_train, y_train, batch_size=32, epochs=5)
    evaluate = model.evaluate(x_test, y_test_cat)
    print(evaluate)