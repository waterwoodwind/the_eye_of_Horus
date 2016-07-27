#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

def y_shadow_list(image_np):
    y_shadow = [0 for y in range(image_np.shape[0])]
    for i in range(len(image_np)):
        for j in range(image_np.shape[1]):
            if image_np[i,j,2]<>0:
                y_shadow[i] = y_shadow[i] + image_np[i,j,2]
    
    return y_shadow

def x_shadow_list(image_np):
    x_shadow = [0 for x in range(image_np.shape[1])]
    
    for j in range(image_np.shape[1]):
        for i in range(image_np.shape[0]):
            if image_np[i,j,2]<>0:
                x_shadow[j] = x_shadow[j] + image_np[i,j,2]
    
    return x_shadow

def shadow_border(shadow_list):
    start_index = []
    end_index = []
    
    for i in range(len(shadow_list) - 1):
        if shadow_list[i] == 0 and shadow_list[i + 1] <> 0:
            start_index.append(i+1)
        if shadow_list[i] <> 0 and shadow_list[i + 1] == 0:
            end_index.append(i)

    return start_index,end_index    

img = cv2.imread('2.bmp')
ret,thresh_img = cv2.threshold(img,208,255,cv2.THRESH_BINARY_INV)

y_s_list = y_shadow_list(thresh_img)
y_start_list, y_end_list = shadow_border(y_s_list)
x_s_list = x_shadow_list(thresh_img)
x_start_list, x_end_list = shadow_border(x_s_list)

line_1 = thresh_img[y_start_list[0]:y_end_list[0]+1,:]
cv2.imshow("line_1", line_1)
cv2.waitKey(0)
x_s_list = x_shadow_list(line_1)
x_start_list, x_end_list = shadow_border(x_s_list)

Character_1 = line_1[:, x_start_list[1]:x_end_list[1]+1]
cv2.imshow("Character_1", Character_1)
cv2.waitKey(0)
cv2.imwrite('Character_2.bmp', Character_1)





