import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def handle(w, h):
    img_data = readFile()
    resized = tf.image.resize(img_data, (w, h)) # 第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法  
    resized = np.asarray(resized.numpy(), dtype='uint8') # 变为uint8才能显示
    plt.imshow(resized)
    plt.show()
    return  resized


def readFile(path):
    #'C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/label.png'
    image_raw_data = tf.io.gfile.GFile(path,
                                       'rb').read()  # 加载原始图像
    img_data = tf.image.decode_jpeg(image_raw_data)  # 解码
    #plt.imshow(img_data.numpy())
    #plt.show()
    return img_data


if __name__ == '__main__':
    train_data = readFile('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/img.png')
    print(train_data.shape)
    tf.image.resize
    print(tr.shape)

    t = tf.reshape(shape=(487,222),tensor=train_data)
    print(t.shape)
    #handle(2, 2)