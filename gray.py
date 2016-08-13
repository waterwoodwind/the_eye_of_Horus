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

zero_data_height_start = 167
zero_data_height_end = 1002

two_flt_width_start = 124
two_flt_width_end = 337
two_plan_lanch_start = 797
two_plan_lanch_end = 850
two_index_start = 3
two_index_end = 43
line_plane_number_start = 93
line_plane_number_end = 132
line_flt_number_start = 1
line_flt_number_end = 60
line_stand_start = 179
line_stand_end = 208
character_width = 9

def clear_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
def clear_dir():
    path_list = ["line", "character", "no_repeat_character"]
    for item in path_list:
        clear_dirs(item)
    
    
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
    
    def shadow_border(self, shadow_list):
        start_index = []
        end_index = []
        
        for i in range(len(shadow_list) - 1):
            if shadow_list[i] == 0 and shadow_list[i + 1] <> 0:
                start_index.append(i+1)
            if shadow_list[i] <> 0 and shadow_list[i + 1] == 0:
                end_index.append(i)
    
        return start_index,end_index  
        
    def get_data_thresh_img(self, img_path):    
        img=cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data_gray_img = gray[zero_data_height_start:zero_data_height_end+1,4:1907+1]
        ret,thresh_img = cv2.threshold(data_gray_img,160,255,cv2.THRESH_BINARY_INV)
        return thresh_img, data_gray_img
    
    def y_x_border_list(self, img):
        y_s_list = self.y_shadow_list(img)
        y_start_list, y_end_list = self.shadow_border(y_s_list)
        x_s_list = self.x_shadow_list(img)
        x_start_list, x_end_list = self.shadow_border(x_s_list)
        return y_start_list, y_end_list, x_start_list, x_end_list         

    def del_surplus_y_line(self, y_start_list, y_end_list):
        #有选中蓝色背景的去除第一个end
        y_end_list.pop(0)
        start_number = len(y_start_list)
        end_number = len(y_end_list)
                
        if start_number > end_number:
            y_start_list.pop()
        for i in range(end_number):
            diff = y_end_list[i] - y_start_list[i] + 1
            print y_end_list[i], y_start_list[i]
            print i, diff
            #当出现数据高度不足时            
            if diff <13:
                y_start_list.pop(i)
                y_end_list.pop(i)
            #当出现14行的字母Q时
            elif diff == 14:
                y_start_list[i] = y_start_list[i] + 1
            #当末尾出现航班的那一行高度太短时
            if y_end_list[i] > 832:
                y_start_list.pop(i)
                y_end_list.pop(i)
        return y_start_list, y_end_list

clear_dir()
character_list = []
img_dir = "multi_img"
#img_dir = "single_img"
for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    img_name = img_file[:-4]
    print img_path
    thresh_img, data_gray_img = Pre_treat().get_data_thresh_img(img_path)
    img_thresh_flt = thresh_img[:,two_flt_width_start:two_flt_width_end + 1]
    data_flt = data_gray_img[:, two_flt_width_start:two_flt_width_end + 1]
    data_plan_arrive = data_gray_img[:, two_plan_lanch_start:two_plan_lanch_end+1]
    #避免边缘的锯齿，缩小边缘
    img_index = thresh_img[:, two_index_start:two_index_end +1]
    
    y_start_list, y_end_list, x_start_list, x_end_list = \
    Pre_treat().y_x_border_list(img_index)
    
    y_start_list, y_end_list = \
    Pre_treat().del_surplus_y_line(y_start_list, y_end_list)
    
    #加入第一行蓝色区域
    y_start_list.insert(0, 7)
    y_end_list.insert(0, 19)
    
    #对第一行蓝色区域进行反色处理
    ret,thresh_img[7-3:19+1+3,:] = cv2.threshold(thresh_img[7-3:19+1+3,:],254,255,cv2.THRESH_BINARY_INV)
    
    for i in range(len(y_start_list)):
        line = img_thresh_flt[y_start_list[i]-3:y_end_list[i]+1+3,:]
        line_color = data_flt[y_start_list[i]-3:y_end_list[i]+1+3,:]
        line_data_plan_arrive = \
        data_plan_arrive[y_start_list[i]-3:y_end_list[i]+1+3,:]
        line_merge = np.hstack((line_color, line_data_plan_arrive))
        cv2.imwrite('line/'+ img_name+ '_line_' + str(i) + '.bmp', line_merge)
        
        
        for j in range(line_flt_number_start, line_flt_number_end + 1,character_width+1):
            Character = line[:, j: j+character_width+1]
            #cv2.imshow("Character_1", Character_1)
            #cv2.waitKey(0)
            if Character.sum()<>0:
                cv2.imwrite('character/'+ img_name+ '_line_' + str(i) + '_' + str(j) + 'flt.tif', Character)
                character_list.append(Character.tolist())
                
        for j in range(line_plane_number_start, line_plane_number_end + 1,character_width+1):
            Character = line[:, j: j+character_width+1]
            #cv2.imshow("Character_1", Character_1)
            #cv2.waitKey(0)
            if Character.sum()<>0:
                cv2.imwrite('character/'+ img_name+ '_line_' + str(i) + '_' + str(j) + 'plane.tif', Character)
                character_list.append(Character.tolist())
                
        for j in range(line_stand_start, line_stand_end + 1,character_width+1):
            Character = line[:, j: j+character_width+1]
            #cv2.imshow("Character_1", Character_1)
            #cv2.waitKey(0)
            if Character.sum()<>0:
                cv2.imwrite('character/'+ img_name+ '_line_' + str(i) + '_' + str(j) + 'plane.tif', Character)
                character_list.append(Character.tolist())

character_series = pd.Series(character_list)
no_repeat = character_series.value_counts()
for number,item in enumerate(no_repeat.index):
    print item, no_repeat.iloc[number]
    np_item = np.array(item)        
    cv2.imwrite('no_repeat_character/' + \
    str(no_repeat.iloc[number]) + '_' + \
    str(number) + '.tif', np_item)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        