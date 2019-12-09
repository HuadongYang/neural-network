import tensorflow as tf
import ImageHandler as hd
from time import time

def getData():
    train_data = hd.readFile('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/img.png')
    label_data = hd.readFile('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/label.png')
    return train_data, label_data
def train(train_data, label_data):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(222, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(222, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(5, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    startTime = time()
    history = model.fit(train_data, label_data, batch_size=50, epochs=2)
    duration = time() - startTime
    print("duration: ", duration)
    plt.plot(history.epoch, history.history.get('accuracy'), label="acc")
    plt.plot(history.epoch, history.history.get('val_accuracy'), label="val_acc")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    train_data, label_data = getData()
    train(train_data, label_data)


