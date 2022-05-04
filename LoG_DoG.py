from random import gauss

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from numba import njit, prange

img = cv2.imread("123.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@njit(fastmath=True, parallel=True)
def gausskernel(sigma):
    size = 3 * sigma
    gausskernel = np.zeros((size, size), np.float32)
    for i in prange(size):
        for j in prange(size):
            norm = math.pow(i - 1, 2) + math.pow(j - 1, 2)
            gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))  # Найти гауссову свертку
    sum = np.sum(gausskernel)  # сумма
    kernel = gausskernel / sum  # нормализация
    return kernel


@njit(fastmath=True, parallel=True)
def gauss_transform(img, sigma):
    kernel = gausskernel(sigma)
    for i in prange(1, img.shape[0] - 1):
        for j in prange(1, img.shape[1] - 1):
            suma = 0
            for k in prange(-1, 2):
                for l in prange(-1, 2):
                    suma += img[i + k, j + l] * kernel[k + 1, l + 1]  # фильтр Гаусса
            img[i, j] = suma
    return img



@njit(fastmath=True, parallel=True)
def DoG(copy_img, high, low):
    for i in prange(copy_img.shape[0]):
        for j in prange(copy_img.shape[1]):
            copy_img[i, j] = 5 * (high[i, j] - low[i, j])
    return copy_img

def log(org, gauss, copy_org, kernel_radius, kernel_x, kernel_y, sigma2: float):
    for y in range(kernel_radius, org.shape[0] - kernel_radius):
        for x in range(kernel_radius, org.shape[1] - kernel_radius):
            dx = 0
            dy = 0

            for i in range(-kernel_radius, kernel_radius + 1):
                for j in range(-kernel_radius, kernel_radius + 1):
                    dx += kernel_x[i][j]*gauss[x-i,y-j]
                    dy += kernel_y[i][j]*gauss[x-i,y-j]
            d = (-1 / (math.pi * pow(sigma2, 4))) * (
                    1 - (pow(dx, 2) + pow(dy, 2)) / 2 * pow(sigma2, 2)) * math.exp(
                -1 * ((pow(dx, 2) + pow(dy, 2)) / 2 * pow(sigma2, 2))) * 480
            if (d==0):
                copy_org[x, y] = d
            else:
                copy_org[x, y] = 255
    return copy_org

dog_img = copy.deepcopy(gray_img)
log_img = copy.deepcopy(gray_img)
ori_img = copy.deepcopy(gray_img)
high_img = copy.deepcopy(gray_img)
low_img = copy.deepcopy(gray_img)
gauss_img = copy.deepcopy(gray_img)
coff = 3
sigma = 1
sigma2 = 1
high = gauss_transform(high_img, coff * sigma)
low = gauss_transform(low_img, sigma)
kernel_radius = int(3 / 2)
kernel_x = cv2.Sobel(src=gauss_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
kernel_y = cv2.Sobel(src=gauss_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
DoG_img = DoG(dog_img, high, low)
LoG_img = log(ori_img, gauss_img, log_img, kernel_radius, kernel_x, kernel_y, sigma2)




# Операции вывода
plt.subplot(231), plt.imshow(gray_img, cmap=plt.get_cmap(name='gray'))
plt.title('gray'), plt.axis('off')
plt.subplot(232), plt.imshow(DoG_img, cmap=plt.get_cmap(name='gray'))
plt.title('DoG'), plt.axis('off')
plt.subplot(233), plt.imshow(high, cmap=plt.get_cmap(name='gray'))
plt.title('high'), plt.axis('off')
plt.subplot(234), plt.imshow(low, cmap=plt.get_cmap(name='gray'))
plt.title('low'), plt.axis('off')
plt.subplot(235), plt.imshow(LoG_img, cmap=plt.get_cmap(name='gray'))
plt.title('LoG'), plt.axis('off')
plt.show()
