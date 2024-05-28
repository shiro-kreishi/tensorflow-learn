import tensorflow as tf
keras = tf.keras
utils = keras.utils
datasets = keras.datasets
K = keras.backend
KL = keras.layers
Lambda, Input, Flatten = KL.Lambda, KL.Input, KL.Flatten
Model = keras.Model

if __name__ == '__main__':
    Sequential = keras.Sequential
    Dense = KL.Dense
    mnist = datasets.mnist
    to_categorical = utils.to_categorical

    # Загрузка и предобработка данных
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Создание модели
    model = Sequential([
        Dense(512, activation='relu', input_shape=(28 * 28,)),
        Dense(10, activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # Оценка модели
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

