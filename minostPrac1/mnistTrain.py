from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from time import time
import labelme


def getData():
    mnist = tf.keras.datasets.mnist
    (train_image, train_lable), (test_image, test_lable) = mnist.load_data()
    plt.imshow(train_image[5])
    plt.show()
    return (train_image, train_lable), (test_image, test_lable)


def preHandleData(train_lable):
    print(train_lable.shape)
    train_lable_onehot = tf.keras.utils.to_categorical(train_lable)
    print(train_lable_onehot.shape)
    return train_lable_onehot


def buildModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(16, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(16, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))


def trainModel( train_image, train_lable_onehot, test_image, text_lable_onehot):
    #train_image = train_image.reshape((60000, 28 * 28))
    #test_image = test_image.reshape((10000, 28 * 28))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(16, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(16, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    startTime = time()
    history = model.fit(train_image, train_lable_onehot, batch_size=100, epochs=2,
                        validation_data=(test_image, text_lable_onehot))
    duration = time() - startTime
    print("duration: ", duration)
    plt.plot(history.epoch, history.history.get('accuracy'), label="acc")
    plt.plot(history.epoch, history.history.get('val_accuracy'), label="val_acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    (train_image, train_lable), (test_image, test_lable) = getData()
    train_lable_onehot = preHandleData(train_lable)
    text_lable_onehot = preHandleData(test_lable)
    trainModel( train_image, train_lable_onehot, test_image, text_lable_onehot)