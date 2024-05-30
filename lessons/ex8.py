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


class NeuralNetwork(tf.keras.layers.Layer):
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
    y = model(tf.constant([[1.0, 2., 3.,]]))
    print(y)