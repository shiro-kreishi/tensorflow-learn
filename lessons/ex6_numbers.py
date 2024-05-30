import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class DenseNN(tf.Module):
    def __init__(self, inputs, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.w = tf.Variable(tf.random.truncated_normal((inputs, outputs), stddev=0.1), name='w')
        self.b = tf.Variable(tf.zeros([outputs], dtype=tf.float32), name='b')

    def __call__(self, x):
        y = x @ self.w + self.b
        if self.activate == "relu":
            y = tf.nn.relu(y)
        elif self.activate == "softmax":
            y = tf.nn.softmax(y)
        return y


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
    x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

    y_train = to_categorical(y_train, 10)

    layer_1 = DenseNN(28 * 28, 128, activate="relu")
    layer_2 = DenseNN(128, 10, activate="softmax")

    def model_predict(x):
        y = layer_1(x)
        y = layer_2(y)
        return y

    cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    opt = tf.optimizers.Adam(learning_rate=0.01)
    BATCH_SIZE = 32
    EPOCHS = 10

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    for n in range(EPOCHS):
        loss = 0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                f_loss = cross_entropy(y_batch, model_predict(x_batch))

            loss += f_loss
            grads = tape.gradient(f_loss, [layer_1.w, layer_1.b, layer_2.w, layer_2.b])
            opt.apply_gradients(zip(grads, [layer_1.w, layer_1.b, layer_2.w, layer_2.b]))
        print(f"Epoch {n+1}, Loss: {loss.numpy()}")

    y = model_predict(x_test)
    y2 = tf.argmax(y, axis=1).numpy()
    acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
    print("Accuracy: {:.2f}%".format(acc))

    # acc = tf.metric.Accuracy()
    # acc.update_state(y_test, y2)
    # print("Accuracy: {:.2f}%".format(acc))



