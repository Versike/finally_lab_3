import cv2
import numpy as np
from cv2 import convertScaleAbs

cap = cv2.VideoCapture('videoplayback.mp4')

while (1):

    # Take each frame
    _, frame = cap.read()
    sobelx3 = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely3 = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x3 = cv2.convertScaleAbs(sobelx3)
    abs_grad_y3 = cv2.convertScaleAbs(sobely3)
    sobelx5 = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely5 = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    abs_grad_x5 = cv2.convertScaleAbs(sobelx5)
    abs_grad_y5 = cv2.convertScaleAbs(sobely5)
    sobelx7 = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=7)
    sobely7 = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=7)
    abs_grad_x7 = cv2.convertScaleAbs(sobelx7)
    abs_grad_y7 = cv2.convertScaleAbs(sobely7)


    grad3 = cv2.addWeighted(abs_grad_x3, 0.5, abs_grad_y3, 0.5, 0)
    grad5 = cv2.addWeighted(abs_grad_x5, 0.5, abs_grad_y5, 0.5, 0)
    grad7 = cv2.addWeighted(abs_grad_x7, 0.5, abs_grad_y7, 0.5, 0)

    low_sigma = cv2.GaussianBlur(frame, (3, 3), 0)
    high_sigma = cv2.GaussianBlur(frame, (5, 5), 0)
    dog = low_sigma - high_sigma
    dog = convertScaleAbs(dog)
    sigma = 1
    gauss = cv2.GaussianBlur(frame, (3, 3), sigma)
    laplacian = cv2.Laplacian(gauss, cv2.CV_64F, ksize = 3)
    laplacian = convertScaleAbs(laplacian)

    cv2.imshow('SOBEL 3X3', grad3)
    cv2.imshow('SOBEL 5X5', grad5)
    cv2.imshow('SOBEL 7X7', grad7)
    cv2.imshow('DIFFOFGAUSSIAN', dog)
    cv2.imshow('laplacian', laplacian)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# release the frame
cap.release()