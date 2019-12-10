from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

def read():
    train_data_dir='C:/Users/yanghd/Desktop/desk/工作/文字识别/download/train_img'

    val_data_dir='C:/Users/yanghd/Desktop/desk/工作/文字识别/download/label_img'

    img_width,img_height = 200, 200

    batch_size=16

    train_datagen=ImageDataGenerator(rescale = 1. / 255,shear_range = 0.2,horizontal_flip = True)

    val_datagen=ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(train_data_dir,target_size = (img_width,img_height),batch_size = batch_size)
    for i in range(3):
        print(train_generator.next())

    val_generator=val_datagen.flow_from_directory(val_data_dir,target_size = (img_width,img_height),batch_size = batch_size)
def readFile(path):
    #'C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/label.png'
    image_raw_data = tf.io.gfile.GFile(path,
                                       'rb').read()  # 加载原始图像
    img_data = tf.image.decode_jpeg(image_raw_data)  # 解码
    return img_data
def handle(path, w, h):
    img_data = readFile(path)
    resized = tf.image.resize(img_data, (w, h)) # 第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法  
    print(resized.shape)
    resized = np.asarray(resized.numpy(), dtype='uint8') # 变为uint8才能显示
    print(resized.shape)
    plt.imshow(resized)
    plt.show()
    return  resized
def batchRead(path):
    file_names = os.listdir(path)
    img_datas = np.empty((2, 200, 200,4))
    i = 0
    for file_name in file_names:
        img_datas[i]=handle(path+'/'+file_name,200,200)
    print(img_datas.shape)

if __name__ == '__main__':
    batchRead('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/train_img')