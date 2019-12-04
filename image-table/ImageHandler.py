import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def handle(w, h):
    image_raw_data = tf.io.gfile.GFile('C:/Users/yanghd/Desktop/desk/工作/文字识别/download/p4.png', 'rb').read()  # 加载原始图像
    img_data = tf.image.decode_jpeg(image_raw_data)# 解码
    plt.imshow(img_data.numpy())
    plt.show()
    tf.image.resize_with_crop_or_pad
    resized = tf.image.resize_with_crop_or_pad(img_data, 64, 64) # 第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法  
    resized = np.asarray(resized.numpy(), dtype='uint8') # 变为uint8才能显示
    plt.imshow(resized)
    plt.show()
if __name__ == '__main__':
    handle(11,12)