import numpy as np
from numba import prange, njit
import cv2 as cv2
import math

@njit(fastmath=True, parallel=True)
def sobel_filter_three_1(origin, adele_new_img):
    matrix_x = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    matrix_y = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    for i in prange(1, adele_new_img.shape[0] - 1):
        for j in prange(1, adele_new_img.shape[1] - 1):
            array = np.zeros((3, 3), dtype="int")
            array[0, 0] = origin[i - 1, j - 1]
            array[1, 0] = origin[i, j - 1]
            array[2, 0] = origin[i + 1, j - 1]

            array[0, 1] = origin[i - 1, j]
            array[1, 1] = origin[i, j]
            array[2, 1] = origin[i + 1, j]

            array[0, 2] = origin[i - 1, j + 1]
            array[1, 2] = origin[i, j + 1]
            array[2, 2] = origin[i + 1, j + 1]

            g_x = 0
            g_y = 0
            for x in prange(3):
                for y in prange(3):
                    g_x += array[x, y] * matrix_x[x, y]
                    g_y += array[x, y] * matrix_y[x, y]

            g = (g_x ** 2 + g_y ** 2) ** 0.5
            if g > 255:
                g = 255
                adele_new_img[i, j] = g
            else:
                adele_new_img[i, j] = g

    return adele_new_img



@njit(fastmath=True, parallel=True)
def sobel_filter_five_1(origin, adele_new_img):
    matrix_x = np.array([[-2, -2, -2, -2, -2],
                         [-1, -1, -1, -1, -1],
                         [0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [2, 2, 2, 2, 2]])

    matrix_y = np.array([[-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2]])

    for i in prange(2, origin.shape[0] - 2):
        for j in prange(2, origin.shape[1] - 2):
            array = np.zeros((5, 5), dtype="int")

            array[0, 0] = origin[i - 2, j - 2]
            array[1, 0] = origin[i - 1, j - 2]
            array[2, 0] = origin[i, j - 2]
            array[3, 0] = origin[i + 1, j - 2]
            array[4, 0] = origin[i + 2, j - 2]

            array[0, 1] = origin[i - 2, j - 1]
            array[1, 1] = origin[i - 1, j - 1]
            array[2, 1] = origin[i, j - 1]
            array[3, 1] = origin[i + 1, j - 1]
            array[4, 1] = origin[i + 2, j - 1]

            array[0, 2] = origin[i - 2, j]
            array[1, 2] = origin[i - 1, j]
            array[2, 2] = origin[i, j]
            array[3, 2] = origin[i + 1, j]
            array[4, 2] = origin[i + 2, j]

            array[0, 3] = origin[i - 2, j + 1]
            array[1, 3] = origin[i - 1, j + 1]
            array[2, 3] = origin[i, j + 1]
            array[3, 3] = origin[i + 1, j + 1]
            array[4, 3] = origin[i + 2, j + 1]

            array[0, 4] = origin[i - 2, j + 2]
            array[1, 4] = origin[i - 1, j + 2]
            array[2, 4] = origin[i, j + 2]
            array[3, 4] = origin[i + 1, j + 2]
            array[4, 4] = origin[i + 2, j + 2]

            g_x = 0
            g_y = 0
            for x in prange(5):
                for y in prange(5):
                    g_x += array[x, y] * matrix_x[x, y]
                    g_y += array[x, y] * matrix_y[x, y]

            g = (g_x ** 2 + g_y ** 2) ** 0.5
            if g > 255:
                g = 255
                adele_new_img[i, j] = g
            else:
                adele_new_img[i, j] = g
    return adele_new_img


@njit(fastmath=True, parallel=True)
def sobel_filter_seven_1(origin, adele_new_img):
    matrix_x = np.array([[-3, -3, -3, -3, -3, -3, -3],
                         [-2, -2, -2, -2, -2, -2, -2],
                         [-1, -1, -1, -1, -1, -1, -1],
                         [0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [2, 2, 2, 2, 2, 2, 2],
                         [3, 3, 3, 3, 3, 3, 3]])

    matrix_y = np.array([[-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3]])

    for i in prange(3, origin.shape[0] - 3):
        for j in prange(3, origin.shape[1] - 3):
            array = np.zeros((7, 7), dtype="int")

            array[0, 0] = origin[i - 3, j - 3]
            array[1, 0] = origin[i - 2, j - 3]
            array[2, 0] = origin[i - 1, j - 3]
            array[3, 0] = origin[i, j - 3]
            array[4, 0] = origin[i + 1, j - 3]
            array[5, 0] = origin[i + 2, j - 3]
            array[6, 0] = origin[i + 3, j - 3]

            array[0, 1] = origin[i - 3, j - 2]
            array[1, 1] = origin[i - 2, j - 2]
            array[2, 1] = origin[i - 1, j - 2]
            array[3, 1] = origin[i, j - 2]
            array[4, 1] = origin[i + 1, j - 2]
            array[5, 1] = origin[i + 2, j - 2]
            array[6, 1] = origin[i + 3, j - 2]

            array[0, 2] = origin[i - 3, j - 1]
            array[1, 2] = origin[i - 2, j - 1]
            array[2, 2] = origin[i - 1, j - 1]
            array[3, 2] = origin[i, j - 1]
            array[4, 2] = origin[i + 1, j - 1]
            array[5, 2] = origin[i + 2, j - 1]
            array[6, 2] = origin[i + 3, j - 1]

            array[0, 3] = origin[i - 3, j]
            array[1, 3] = origin[i - 2, j]
            array[2, 3] = origin[i - 1, j]
            array[3, 3] = origin[i, j]
            array[4, 3] = origin[i + 1, j]
            array[5, 3] = origin[i + 2, j]
            array[6, 3] = origin[i + 3, j]

            array[0, 4] = origin[i - 3, j + 1]
            array[1, 4] = origin[i - 2, j + 1]
            array[2, 4] = origin[i - 1, j + 1]
            array[3, 4] = origin[i, j + 1]
            array[4, 4] = origin[i + 1, j + 1]
            array[5, 4] = origin[i + 2, j + 1]
            array[6, 4] = origin[i + 3, j + 1]

            array[0, 5] = origin[i - 3, j + 2]
            array[1, 5] = origin[i - 2, j + 2]
            array[2, 5] = origin[i - 1, j + 2]
            array[3, 5] = origin[i, j + 2]
            array[4, 5] = origin[i + 1, j + 2]
            array[5, 5] = origin[i + 2, j + 2]
            array[6, 5] = origin[i + 3, j + 2]

            array[0, 6] = origin[i - 3, j + 3]
            array[1, 6] = origin[i - 2, j + 3]
            array[2, 6] = origin[i - 1, j + 3]
            array[3, 6] = origin[i, j + 3]
            array[4, 6] = origin[i + 1, j + 3]
            array[5, 6] = origin[i + 2, j + 3]
            array[6, 6] = origin[i + 3, j + 3]

            g_x = 0
            g_y = 0
            for x in prange(7):
                for y in prange(7):
                    g_x += array[x, y] * matrix_x[x, y]
                    g_y += array[x, y] * matrix_y[x, y]

            g = (g_x ** 2 + g_y ** 2) ** 0.5
            if g > 255:
                g = 255
                adele_new_img[i, j] = g
            else:
                adele_new_img[i, j] = g
    return adele_new_img