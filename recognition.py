# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 18:51:43 2016

@author: Administrator
"""

import cv2
import os
import numpy as np
from gray import Pre_treat
from gray import header


import pandas as pd
import numpy as np

import shutil

zero_data_height_start = 167
zero_data_height_end = 1002

two_flt_width_start = 124
two_flt_width_end = 337
two_plan_lanch_start = 805
two_plan_lanch_end = 860
two_index_start = 3
two_index_end = 43
line_plane_number_start = 93
line_plane_number_end = 132
line_flt_number_start = 1
line_flt_number_end = 60
line_stand_start = 179
line_stand_end = 208
line_data_plan_arrive_start = 9
line_data_plan_arrive_end = 48
character_width = 9


class Recognise(object):
    def __init__(self):
        train_main_path = 'train_data'
        train_version_path = '2016-08-17_08-26'
        self.dir_path = os.path.join(train_main_path, train_version_path)


    def get_train_path(self):
        train_main_path = 'train_data'
        train_version_path = '2016-08-13 18-39'
        dir_path = os.path.join(train_main_path, train_version_path)
        return dir_path

    def load_data(self, dir_path):
        list_img = []
        list_digits_target = []
        recognition_dict = {}
        for digit_name in os.listdir(dir_path):
            digit_path = os.path.join(dir_path, digit_name)
            img = cv2.imread(digit_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_one = gray.reshape(-1)
            list_gray_one = gray_one.tolist()
            str_gray_one = str(list_gray_one)
            list_img.append(str_gray_one)
            list_digits_target.append(digit_name[-5:-4])
            recognition_dict[str_gray_one] = digit_name[-5:-4]


        return recognition_dict

    def np_to_digit(self, line, start, end):
        recg_dict = self.load_data(self.dir_path)
        str_data = ''
        for j in range(start, end,character_width+1):
            Character = line[:, j: j+character_width+1]
            #cv2.imshow("Character_1", Character_1)
            #cv2.waitKey(0)
            if Character.sum()<>0:
                Character_one = Character.reshape(-1)
                list_Character_one = Character_one.tolist()
                str_Character_one = str(list_Character_one)
                str_data = str_data + recg_dict[str_Character_one]
        return str_data
        
if __name__ == '__main__':
    train_path = Recognise().get_train_path()
    recognition_dict = Recognise().load_data(train_path)
    
    character_list = []
    img_dir = "multi_img"
    #img_dir = "single_img"
    list_flt_data = []
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        print img_path
        #最原始的图
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算预达、变更到达的序号
        plan_arrive_start = header().plan_arrive_start(gray)
        #含航班数据的图
        thresh_img, data_gray_img = Pre_treat().get_data_thresh_img(img_path)
        img_thresh_flt = thresh_img[:, two_flt_width_start:two_flt_width_end + 1]
        img_thresh_plan_arrive = thresh_img[:, plan_arrive_start:plan_arrive_start+44]
        data_flt = data_gray_img[:, two_flt_width_start:two_flt_width_end + 1]
        data_plan_arrive = data_gray_img[:, plan_arrive_start:plan_arrive_start+44]
        
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
            line_thresh_arrive = img_thresh_plan_arrive[y_start_list[i]-3:y_end_list[i]+1+3,:]
            
            line_color = data_flt[y_start_list[i]-3:y_end_list[i]+1+3,:]
            line_data_plan_arrive = \
            data_plan_arrive[y_start_list[i]-3:y_end_list[i]+1+3,:]
            line_merge = np.hstack((line_color, line_data_plan_arrive))
            
            
            
            str_flt_number = Recognise().np_to_digit(recognition_dict,
                                         line,
                                         line_flt_number_start, 
                                         line_flt_number_end + 1)
                                         
            
            str_plane_number = Recognise().np_to_digit(recognition_dict,
                                         line,
                                         line_plane_number_start, 
                                         line_plane_number_end + 1)
                
                    
            str_stand_number =Recognise().np_to_digit(recognition_dict,
                                         line,
                                         line_stand_start, 
                                         line_stand_end + 1)
                                         
            str_plan_arrive_number = Recognise().np_to_digit(recognition_dict,
                                                 line_thresh_arrive,
                                                 0, 
                                                 39 + 1)
            
            list_flt_data.append([str_flt_number, 
                                  str_plane_number, 
                                  str_stand_number, 
                                  str_plan_arrive_number])            
            
            print str_flt_number, str_plane_number, str_stand_number, str_plan_arrive_number
    
    df_flt_data = pd.DataFrame(list_flt_data)
    df_flt_data.to_csv(u'航班号_机号_机位.csv', encoding= 'utf-8', header=False, index=False)
    df_flt_data.to_excel(u'航班号_机号_机位.xlsx', encoding= 'utf-8', header=False, index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    