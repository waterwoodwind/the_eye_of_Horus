# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:03:53 2016

@author: Administrator
"""
import cv2
import os
import pandas as pd
import numpy as np
import shutil
from matplotlib import pyplot as plt

def show_img(img_name):
    cv2.imshow('image',img_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    img_dir = "image"
    for img_file in os.listdir(img_dir):
        print img_file
        img_path = os.path.join(img_dir, img_file)
        img_name = img_file[:-4]
        print img_path
        #最原始的图
        img = cv2.imread(img_path)
        #灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #裁剪得wps数据区域
        data_img = gray_img[164:991,30:1893]
        #二值化 反转        
        ret,thresh_img = cv2.threshold(data_img,160,255,cv2.THRESH_BINARY_INV)
        plt.imshow(thresh_img, cmap = 'gray',interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        cv2.imwrite('thresh_img.png', thresh_img)
        