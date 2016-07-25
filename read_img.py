#coding=utf-8
import cv2
import numpy as np

img = cv2.imread('2.bmp')
px = img[8, 20]
print px

ret,thresh1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,254,255,cv2.THRESH_BINARY)
ret,thresh3 = cv2.threshold(img,208,255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshold", thresh3)
cv2.waitKey(0)