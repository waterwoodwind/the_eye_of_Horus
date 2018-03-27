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
    ret,thresh_img = cv2.threshold(gray_img,160,255,cv2.THRESH_BINARY_INV)
    #裁剪得wps表格数据区域
    excel_img = thresh_img[164:991,30:1893]
    #plt.imshow(data_img, cmap = 'gray',interpolation = 'bicubic')
    cv2.imwrite(img_file, excel_img)
    
    # 搜索获取实际数据区域
    x_shadow_list = Pre_treat().x_shadow_list(excel_img)
    y_shadow_list = Pre_treat().y_shadow_list(excel_img)
    #plt.bar(range(len(x_shadow_list)), x_shadow_list)
    #plt.show
    ##获取有数据区域的上下界
    x_begin,x_end = Pre_treat().search_range(x_shadow_list)  
    y_begin,y_end = Pre_treat().search_range(y_shadow_list)
    data_img = excel_img[y_begin:y_end+1, x_begin:x_end+1]
    cv2.imwrite("data_img.bmp", data_img)
    
    
    # 取出表格
    ## 取出水平线，先腐蚀再膨胀
    kernel=np.uint8(np.zeros((23,23)))  
    for x in range(23):  
        kernel[11,x]=255;
    #腐蚀图像   
    horizon_eroded = cv2.erode(data_img,kernel); 
    #膨胀图像    
    horizon_dilated = cv2.dilate(horizon_eroded,kernel)
    
    ## 取出垂直线，先腐蚀再膨胀
    vertical_kernel=np.uint8(np.zeros((23,23)))  
    for x in range(23):  
        vertical_kernel[x,11]=255;
    #腐蚀图像   
    vertical_eroded = cv2.erode(data_img,vertical_kernel); 
    #膨胀图像    
    vertical_dilated = cv2.dilate(vertical_eroded,vertical_kernel)
    
    #得到表格网格线
    mask = cv2.add(horizon_dilated, vertical_dilated)
    cv2.imwrite("mask.bmp", mask)
    
    #得到交点图
    joints = cv2.bitwise_and(horizon_dilated, vertical_dilated)
    cv2.imwrite("joints.bmp", joints)        
    

    #划分单元格
    #本例情况，无合并单元格，无左界，无上界，底部可能有无下界的残余
    horizon_shadow_list = Pre_treat().y_shadow_list(joints)
    mete_up_list = []
    mete_down_list = []
    up_border = 0
    i = 1
    for index, item in enumerate(horizon_shadow_list):
        if item == 0:
            continue
        else:
            down_border = index - 1 + 1
            line = data_img[up_border: down_border]
            cv2.imwrite('line/'+ img_name+ '_line_' + str(i) + '.bmp', line)
            mete_up_list.append(up_border)
            mete_down_list.append(down_border)
            up_border = index + 1
            i = i + 1
            
    vertical_shadow_list = Pre_treat().x_shadow_list(joints)
    mete_left_list = []
    mete_right_list = []
    left_border = 0
    i = 1
    for index, item in enumerate(vertical_shadow_list):
        if item == 0:
            continue
        else:
            right_border = index - 1 + 1
            column = data_img[0:int(data_img.shape[1]),left_border: right_border]
            cv2.imwrite('column/'+ img_name+ '_column_' + str(i) + '.bmp', column)
            mete_left_list.append(left_border)
            mete_right_list.append(right_border)
            left_border = index + 1
            i = i + 1
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    