import math
import numpy as np
import cv2 as cv
from numba import prange, njit

@njit(fastmath=True, parallel=True)
def log(origin, sigma, copy_org, LoG, kernel_rad, blur):

    for x in prange(-kernel_rad - 1, kernel_rad + 1):
        for y in prange(-kernel_rad - 1, kernel_rad + 1):
            LoG[kernel_rad + x][kernel_rad + y] = (-1 / (math.pi * pow(sigma, 4))) * (
                    1 - (pow(x, 2) + pow(y, 2)) / 2 * pow(sigma, 2)) * math.exp(
                -1 * ((pow(x, 2) + pow(y, 2)) / 2 * pow(sigma, 2))) * 100
            # print(LoG[kernel_rad + x][kernel_rad + y])
    for y in prange(kernel_rad, origin.shape[0] - kernel_rad):
        for x in prange(kernel_rad, origin.shape[0] - kernel_rad):
            sm = 0
            for i in range(-kernel_rad, kernel_rad + 1):
                for j in range(-kernel_rad, kernel_rad + 1):
                    sm += blur[x - i, y - j] * LoG[i][j]
            copy_org[x, y] = sm

    return copy_org

@njit(fastmath=True, parallel=True)
def DoG(copy_img, high, low):
    for i in prange(copy_img.shape[0]):
        for j in prange(copy_img.shape[1]):
            copy_img[i, j] = 5 * (high[i, j] - low[i, j])
    return copy_img

@njit(fastmath=True, parallel=True)
def delta(actual, original, height, width):
    suma = 0
    for i in prange(height):
        for j in prange(width):
            suma += int(actual[i, j]) - int(original[i, j])
    size = height * width
    Delta = (suma / size)
    return Delta


def mse(actual, original, height, width):
    squared_sum = 0
    for i in prange(height):
        for j in prange(width):
            squared_sum += math.pow(int(original[i, j]) - int(actual[i, j]), 2)
    size = height * width
    mse = (squared_sum / size)
    return mse


def msad(actual, original, height, width):
    suma = 0
    for i in prange(height):
        for j in prange(width):
            suma += abs(int(original[i, j]) - int(actual[i, j]))
    size = height * width
    msad = (suma / size)
    return msad
