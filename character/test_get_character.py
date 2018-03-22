# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:54:44 2018

@author: wangxin
"""

import cv2
import os
import pandas as pd
import numpy as np
import shutil
import copy
from matplotlib import pyplot as plt

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList,Title,Xlabel,Ylabel):
    plt.hist(myList,len(myList))
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.show()
    
class Pre_treat(object):
    
    def __init__(self):
        pass
    
    def y_shadow_list(self, image_np):
        y_shadow = [0 for y in range(image_np.shape[0])]
        for i in range(len(image_np)):
            for j in range(image_np.shape[1]):
                if image_np[i,j]<>0:
                    y_shadow[i] = y_shadow[i] + image_np[i,j]
           
        return y_shadow
    
    def x_shadow_list(self, image_np):
        x_shadow = [0 for x in range(image_np.shape[1])]
        
        for j in range(image_np.shape[1]):
            for i in range(image_np.shape[0]):
                if image_np[i,j]<>0:
                    x_shadow[j] = x_shadow[j] + image_np[i,j]
        
        return x_shadow
    
    def search_range(self, in_shadow_list):
        shadow_list = copy.deepcopy(in_shadow_list)
        #非零的数归一化
        for index, item in enumerate(shadow_list):
            if item<>0:
                shadow_list[index] = 1 
        if 0 in shadow_list:
            ls2 = [str(i) for i in shadow_list]
            nonzero_string= ''.join(ls2)
            begin_index = nonzero_string.find('1')
            end_index = nonzero_string.rfind('1')
        else:
            begin_index = 0
            end_index = len(shadow_list)
        return begin_index, end_index
        
    
if __name__ == '__main__':
    d = os.path.dirname(__file__)
    print d
    parent_path = os.path.dirname(d)
    print parent_path
    img_dir = os.path.join(parent_path, "multi_img")
    print img_dir
    img_file = "5.bmp"
    img_path = os.path.join(img_dir, img_file)
    img_name = img_file[:-4]
    print img_name
    #最原始的图
    img = cv2.imread(img_path)
    #灰度化
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #二值化 反转        
    ret,thresh_img = cv2.threshold(gray_img,160,1,cv2.THRESH_BINARY_INV)
    #裁剪得wps表格数据区域
    excel_img = thresh_img[164:991,30:1893]
    plt.imshow(data_img, cmap = 'gray',interpolation = 'bicubic')
    cv2.imwrite(img_file, excel_img)
    
    # 搜索获取实际数据区域
    x_shadow_list = Pre_treat().x_shadow_list(excel_img)
    y_shadow_list = Pre_treat().y_shadow_list(excel_img)
    plt.bar(range(len(x_shadow_list)), x_shadow_list)
    plt.show
    ##获取有数据区域的上下界
    x_begin,x_end = Pre_treat().search_range(x_shadow_list)  
    y_begin,y_end = Pre_treat().search_range(y_shadow_list)
    data_img = excel_img[y_begin:y_end+1, x_begin:x_end+1]
   
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    