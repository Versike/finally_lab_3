import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit, prange

img = cv2.imread("123.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def convolution1D(img, kernel, x, y, size):
    _y = y - 1
    pixel = 0
    for i in range(size):
        _x = x - 1
        for j in range(size):
            if (_x > -1 and _y > -1 and _x < img.shape[0] and _y < img.shape[1]):
                pix = img[_x, _y]
            else:
                pix = 0
            pixel += (float(kernel[i][j]) * pix)
            _x = _x + 1
        _y = _y + 1
    return pixel

@njit(fastmath=True, parallel=True)
def convolution1D_par(img, kernel, x, y, size):
    _y = y - 1
    pixel = 0
    for i in prange(size):
        _x = x - 1
        for j in prange(size):
            if (_x > -1 and _y > -1 and _x < img.shape[0] and _y < img.shape[1]):
                pix = img[_x, _y]
            else:
                pix = 0
            pixel += (float(kernel[i][j]) * pix)
            _x = _x + 1
        _y = _y + 1
    return pixel



def sobel3x3(image):
    height = image.shape[0]
    width = image.shape[1]

    xfilter = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    yfilter = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    sobel = np.zeros(img.shape)
    for x in range(0, height):
        for y in range(0, width):
            pixel_x = convolution1D(image, xfilter, x, y, 3)
            pixel_y = convolution1D(image, yfilter, x, y, 3)
            val = math.sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y))
            sobel[x, y] = val
    return sobel

@njit(fastmath=True, parallel=True)
def sobel5x5(image):
    height = image.shape[0]
    width = image.shape[1]

    xfilter = np.array([[-1, -1, 0, 1, 1],
                        [-1, -1, 0, 1, 1],
                        [-2, -2, 0, 2, 2],
                        [-1, -1, 0, 1, 1],
                        [-1, -1, 0, 1, 1]])
    yfilter = np.array([[1, 1, 2, 1, 1],
                        [1, 1, 2, 1, 1],
                        [0, 0, 0, 0, 0],
                        [-1, -1, -2, -1, -1],
                        [-1, -1, -2, -1, -1]])

    sobel = np.zeros(img.shape)
    for x in prange(0, height):
        for y in prange(0, width):
            pixel_x = convolution1D_par(image, xfilter, x, y, 5)
            pixel_y = convolution1D_par(image, yfilter, x, y, 5)
            val = math.sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y))
            sobel[x, y] = val
    return sobel

@njit(fastmath=True, parallel=True)
def sobel7x7(image):
    height = image.shape[0]
    width = image.shape[1]

    xfilter = np.array([[-1, -1, -1, 0, 1, 1, 1],
                        [-1, -1, -1, 0, 1, 1, 1],
                        [-1, -1, -1, 0, 1, 1, 1],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-1, -1, -1, 0, 1, 1, 1],
                        [-1, -1, -1, 0, 1, 1, 1],
                        [-1, -1, -1, 0, 1, 1, 1]])
    yfilter = np.array([[1, 1, 1, 2, 1, 1, 1],
                        [1, 1, 1, 2, 1, 1, 1],
                        [1, 1, 1, 2, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [-1, -1, -1, -2, -1, -1, -1],
                        [-1, -1, -1, -2, -1, -1, -1],
                        [-1, -1, -1, -2, -1, -1, -1]])

    sobel = np.zeros(img.shape)
    for x in prange(0, height):
        for y in prange(0, width):
            pixel_x = convolution1D_par(image, xfilter, x, y, 7)
            pixel_y = convolution1D_par(image, yfilter, x, y, 7)
            val = math.sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y))
            sobel[x, y] = val
    return sobel


sobel_img = np.matrix(gray_img)
sobel_img = sobel_img / 255.0


#Операции вывода
plt.subplot(231), plt.imshow(gray_img, cmap=plt.get_cmap(name='gray'))
plt.title('gray'), plt.axis('off')
plt.subplot(232), plt.imshow(sobel3x3(sobel_img), cmap=plt.get_cmap(name='gray'))
plt.title('sobel3x3'), plt.axis('off')
plt.subplot(233), plt.imshow(sobel5x5(sobel_img), cmap=plt.get_cmap(name='gray'))
plt.title('sobel5x5'), plt.axis('off')
plt.subplot(234), plt.imshow(sobel7x7(sobel_img), cmap=plt.get_cmap(name='gray'))
plt.title('sobel7x7'), plt.axis('off')
plt.show()
