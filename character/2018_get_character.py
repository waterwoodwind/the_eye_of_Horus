# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:54:44 2018

@author: wangxin
train_data:0403
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
    
#从单元格中切分出数字字母符号    
def cut_character(chr_img, img_name):    
    chr_list = []
    sinogram_list = []
    sinogram_width = 13
    chr_img_start = 2
    chr_img_end = chr_img.shape[1]
    chr_width = 7
    j = chr_img_start
    recg_dict = Recognise().load_data()
    while j<chr_img_end:
        Character = chr_img[3:16, j: j+chr_width]
        #cv2.imshow("Character_1", Character_1)
        #cv2.waitKey(0)img_character
        if Character.sum()<>0:
            if recg_dict.has_key(str(Character.reshape(-1).tolist())):
                cv2.imwrite('character/'+ img_name+ '_line_' + str(i) + '_' + str(j) + '.tif', Character)
                chr_list.append(Character.tolist())
                j = j + chr_width
            else:
                sinogram = chr_img[3:16, j:j+sinogram_width]
                cv2.imwrite('sinogram/'+ img_name+ '_line_' + str(i) + '_' + str(j) + '.tif', sinogram)
                sinogram_list.append(sinogram.tolist())
                j = j + sinogram_width
        else:
            j = j + chr_width
    return chr_list, sinogram_list

#将不重复字符图片拆出并保存
def get_and_save_no_repeat(character_list):
    character_series = pd.Series(character_list)
    no_repeat = character_series.value_counts()
    for number,item in enumerate(no_repeat.index):
        np_item = np.array(item)        
        cv2.imwrite('no_repeat_character/' + \
        str(no_repeat.iloc[number]) + '_' + \
        str(number) + '.tif', np_item)
        
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

class Recognise(object):
    def __init__(self):
        d = os.path.dirname(__file__)
        parent_path = os.path.dirname(d)
        train_version_path = 'train_data/no_repeat_0403'
        self.dir_path = os.path.join(parent_path, train_version_path)


    def get_train_path(self):
        d = os.path.dirname(__file__)
        parent_path = os.path.dirname(d)
        train_version_path = 'train_data/no_repeat_0403'
        dir_path = os.path.join(parent_path, train_version_path)
        return dir_path

    def load_data(self):
        list_img = []
        list_digits_target = []
        recognition_dict = {}
        for digit_file in os.listdir(self.dir_path):
            digit_path = os.path.join(self.dir_path, digit_file)
            img = cv2.imread(digit_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_one = gray.reshape(-1)
            list_gray_one = gray_one.tolist()
            str_gray_one = str(list_gray_one)
            list_img.append(str_gray_one)
            digit_name = digit_file[-5:-4]
            list_digits_target.append(digit_name)
            if digit_name == "!":
                digit_name = "|"
            if digit_name == ";":
                digit_name = ":"
            recognition_dict[str_gray_one] = digit_name
        return recognition_dict

    def np_to_digit(self, chr_img):
        recg_dict = self.load_data()
        str_data = ''
        chr_img_start = 2
        chr_img_end = chr_img.shape[1]
        chr_width = 7
        for j in range(chr_img_start, chr_img_end + 1,chr_width):
            Character = chr_img[3:16, j: j+chr_width]
            #cv2.imshow("Character_1", Character_1)
            #cv2.waitKey(0)img_character
            if Character.sum()<>0:
                Character_one = Character.reshape(-1)
                list_Character_one = Character_one.tolist()
                str_Character_one = str(list_Character_one)
                try:
                    str_data = str_data + recg_dict[str_Character_one]
                except:
                    str_data = str_data + u"汉"
        return str_data
       
    
if __name__ == '__main__':
    d = os.path.dirname(__file__)
    print d
    parent_path = os.path.dirname(d)
    print parent_path
    img_dir = os.path.join(parent_path, "multi_img/2018")
    print img_dir
    character_list = []
    chinese_list = []
    cell_data_list = []
    for img_file in os.listdir(img_dir):
    
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
        excel_img = thresh_img[164:991,30:1325]
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
                
        cell_img_list = []
        line_img_list = []
        for row,row_item in enumerate(mete_up_list):
            line_img_list = []
            for col, col_item in enumerate(mete_left_list):
                cell_pic = data_img[mete_up_list[row]:mete_down_list[row],\
                                    mete_left_list[col]:mete_right_list[col]]
                cv2.imwrite('cell/' + img_name + '_cell_' + str(row)+ '_' + str(col) + '.bmp', cell_pic)
                line_img_list.append(cell_pic)
                e_list, s_list = cut_character(cell_pic, img_name)
                character_list.extend(e_list)
                chinese_list.extend(s_list)
            cell_img_list.append(line_img_list)
        
        #将含字图片识别为文字
        for row,row_item in enumerate(cell_img_list):
            line_data_list = []
            for col, col_item in enumerate(row_item):
                cell_data = Recognise().np_to_digit(col_item)
                line_data_list.append(cell_data)
            cell_data_list.append(line_data_list)
    
    #将不重复字符图片拆出并保存
    get_and_save_no_repeat(chinese_list)
    
    #df_flt_data = pd.DataFrame(cell_data_list)
    #df_flt_data.to_csv(u'航班号_机号.csv', encoding= 'utf-8', header=False, index=False)
    #df_flt_data.to_excel(u'航班号_机号.xlsx', encoding= 'utf-8', header=False, index=False)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        