import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os

def handle(path, w, h):
    img_data = readFile(path)
    resized = tf.image.resize(img_data, (w, h)) # 第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法  
    print(resized.shape)
    resized = np.asarray(resized.numpy(), dtype='uint8') # 变为uint8才能显示
    print(resized.shape)
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
def skimageRead(path):
    img_data = io.imread(path)
    return img_data

def picread(filelist):
    """
    读取狗的图片并转换成张量
    :param filelist: 文件路f径+名字的列表
    :return: 每张图片的张量
    """
    # 1.构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2.构造阅读器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)

    # 3.对读取的图片进行解码
    image = tf.image.decode_jpeg(value)

    # 4.处理图片的大小（统一大小）
    image_resize = tf.image.resize_images(image,[200,200])

    # 注意：一定要把样本的形状固定，在批处理中要求所有数据的形状必须固定
    image_resize.set_shape([200,200,4])


    # 5.进行批处理
    image_resize_batch = tf.train.batch([image_resize],batch_size=2,num_threads=1,capacity=3)


    return   image_resize


if __name__ == '__main__':
    #train_data = readFile('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/label.png')
    #img_data = skimageRead('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p2_json/img.png')
    #print(train_data.shape)
    #flattern = train_data.numpy().reshape((222*487*3))
    #handle('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p4_json/label.png', 300,300)
    path = 'C:/Users/yanghd/Desktop/desk/工作/文字识别/download/train_img'
    file_name = os.listdir(path)
    #filelist = []
    #for file in file_name:
    #    filelist.append(os.path.join(file_name, file))
    filelist = [os.path.join(path, file) for file in file_name]
    image_batch = picread(filelist)
    print(image_batch.shape)