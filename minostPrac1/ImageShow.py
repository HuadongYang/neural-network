import matplotlib.pyplot as plt

ma = [[1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 0.96078431,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.32156863, 0.75294118, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.42745098, 0.2, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.48235294, 0.67843137, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.71372549, 0.2, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.83529412, 0.50980392, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 0.99607843,
       0.11764706, 0.11764706, 0.8, 0.2, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.83529412, 0.30196078, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.83529412, 0.18431373, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.43921569, 0.72941176, 0.19215686, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.83529412, 0.15686275, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.17647059,
       0.56078431, 0.81176471, 0.36470588, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.83137255, 0.68627451, 0.67058824,
       0.83137255, 0.83529412, 0.78039216, 0.63529412, 0.36470588,
       0.1372549, 0.75686275, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.41568627, 0.56078431, 0.56078431,
       0.56078431, 0.38039216, 0.11764706, 0.11764706, 0.11764706,
       0.38039216, 0.83529412, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.5254902, 0.83529412, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.5254902, 0.83921569, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.5254902, 0.83529412, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.5254902, 0.84313725, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.36862745, 0.83529412, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 0.11764706, 0.11764706, 0.11764706,
       0.11764706, 0.11764706, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       0.07058824, 0.07058824, 0.07058824, 0.07058824, 0.07058824,
       0.07058824, 0.07058824, 0.07058824, 0.07058824, 0.07058824,
       0.07058824, 0.07058824, 0.07058824, 0.07058824, 0.07058824,
       0.07058824, 0.07058824, 1.0, 1.0, 1.0,
       1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0]]
print(ma)
plt.imshow(ma)
plt.show()
