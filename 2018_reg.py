# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 14:04:37 2018

@author: wangxin
"""

import cv2
import os
import pandas as pd
import numpy as np
import shutil
import copy
import time
from matplotlib import pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class Pre_treat(object):
    def __init__(self):
        pass
    
    def local_dir(self):
        img_dir = "multi_img/2018"
        return img_dir
    
    def gray_thresh_255(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, im_at_fixed = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        return im_at_fixed

class Shadow(object):
    
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

class Cut(object):
    def __init__(self):
        pass
    
    def excel(self, img, excel_y_begin, excel_x_begin):
        result_img = img[excel_y_begin:991,excel_x_begin:1325]
        return result_img
    
    def contain(self, img):
        # 搜索获取实际数据区域
        x_shadow_list = Shadow().x_shadow_list(img)
        y_shadow_list = Shadow().y_shadow_list(img)
        #plt.bar(range(len(x_shadow_list)), x_shadow_list)
        #plt.show
        ##获取有数据区域的上下界
        x_begin,x_end = Shadow().search_range(x_shadow_list)  
        y_begin,y_end = Shadow().search_range(y_shadow_list)
        data_img = img[y_begin:y_end+1, x_begin:x_end+1]
        return data_img
    
    def cell(self, data_img):
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
        horizon_shadow_list = Shadow().y_shadow_list(joints)
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
                
        vertical_shadow_list = Shadow().x_shadow_list(joints)
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
            cell_img_list.append(line_img_list)
        return cell_img_list
    
    def img_to_str(self, im_at_fixed):
        im_at_fixed_one = im_at_fixed.reshape(-1)
        list_thresh_one = im_at_fixed_one.tolist()
        for index, item in enumerate(list_thresh_one):
            if item == 255:
                list_thresh_one[index] = 1
        list_thresh_one = [str(list_thresh_one[i]) for i in range(len(list_thresh_one))]
        str_gray_one = "".join(list_thresh_one)
        return str_gray_one
    
    #从单元格中切分出数字字母符号    
    def cut_character(self, chr_img):    
        chr_list = []
        sinogram_width = 13
        chr_img_start = 2
        chr_img_end = chr_img.shape[1]
        chr_width = 7
        j = chr_img_start
        recg_dict = Recognise().load_data()
        while j<chr_img_end:
            Character = chr_img[2:16, j: j+chr_width]
            #cv2.imshow("Character_1", Character_1)
            #cv2.waitKey(0)img_character
            if Character.sum()<>0:
                chr_key = self.img_to_str(Character)
                if recg_dict.has_key(chr_key):
                    #cv2.imwrite('character/'+ img_name+ '_line_' + str(i) + '_' + str(j) + '.tif', Character)
                    chr_list.append(chr_key)
                    j = j + chr_width
                else:
                    sinogram = chr_img[2:16, j:j+sinogram_width]
                    sinogram_key = Recognise().img_to_str(sinogram)
                    chr_list.append(sinogram_key)
                    j = j + sinogram_width
            else:
                j = j + chr_width
        return chr_list
    
    def character(self, cell_img_list):
        result_list = []
        for row,row_item in enumerate(cell_img_list):
            cell_str_list = []
            for col, cell_img in enumerate(row_item):
                chr_list = self.cut_character(cell_img)
                cell_str_list.append(chr_list)
            result_list.append(cell_str_list)
        return result_list
            
class Recognise(object):
    def __init__(self):
        self.train_dir_path = 'character library/ascii'
        self.train_chinese_dir_path = 'character library/chinese'
        
    def img_to_str(self, im_at_fixed):
        im_at_fixed_one = im_at_fixed.reshape(-1)
        list_thresh_one = im_at_fixed_one.tolist()
        for index, item in enumerate(list_thresh_one):
            if item == 255:
                list_thresh_one[index] = 1
        list_thresh_one = [str(list_thresh_one[i]) for i in range(len(list_thresh_one))]
        str_gray_one = "".join(list_thresh_one)
        return str_gray_one
    
    def load_data(self):
        list_img = []
        list_digits_target = []
        recognition_dict = {}
        for digit_file in os.listdir(self.train_dir_path):
            digit_path = os.path.join(self.train_dir_path, digit_file)
            img = cv2.imread(digit_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, im_at_fixed = cv2.threshold(gray, 160, 1, cv2.THRESH_BINARY)
            str_gray_one = self.img_to_str(im_at_fixed)
            list_img.append(str_gray_one)
            digit_name = digit_file[:-4]
            digit_name = unichr(int(digit_name))
            list_digits_target.append(digit_name)
            recognition_dict[str_gray_one] = digit_name
        return recognition_dict
    
    def load_chinese_data(self):
        list_img = []
        list_digits_target = []
        recognition_dict = {}
        for digit_file in os.listdir(self.train_chinese_dir_path):
            digit_path = os.path.join(self.train_chinese_dir_path, digit_file)
            img = cv2.imread(digit_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, im_at_fixed = cv2.threshold(gray, 160, 1, cv2.THRESH_BINARY)
            str_gray_one = self.img_to_str(im_at_fixed)
            list_img.append(str_gray_one)
            digit_name = digit_file[:-4]
            digit_name = unichr(int(digit_name))
            list_digits_target.append(digit_name)
            recognition_dict[str_gray_one] = digit_name
        return recognition_dict
    
    def bin_to_text(self, bin_data_list):
        recg_dict = self.load_data()
        recg_ch_dict = self.load_chinese_data()
        result_list = []
        for row, row_item in enumerate(bin_data_list):
            row_list = []
            for cell,cell_item in enumerate(row_item):
                str_data = ""
                for bin_data in cell_item:
                    if recg_dict.has_key(bin_data):
                        str_data = str_data + recg_dict[bin_data]
                    else:
                        try:
                            str_data = str_data + recg_ch_dict[bin_data]
                        except:
                            str_data = str_data + u'汉'
                row_list.append(str_data)
            result_list.append(row_list)
        return result_list

#去掉东航空客机型        
def remove_mu_a(result_list):
    flt_list = copy.deepcopy(result_list)
    count = 0
    for index in range(len(flt_list)-1,-1,-1):
        row_item = flt_list[index]
        print index, row_item[5]
        if row_item[1].find(u'MU')<>-1 or row_item[2].find(u'MU')<>-1:
            if  row_item[5].find(u'B')==-1:
                count = count + 1
                print index, count, row_item[5]
                flt_list.pop(index)
    return flt_list
                
if __name__ == '__main__':
    img_dir = Pre_treat().local_dir()
    result_list = []
    print "begin"
    loc_file = open(u'C:/Users/Administrator/Desktop/航班数据采集/坐标参数.txt')
    loc_txt = loc_file.read()
    loc_file.close()
    loc_arr = loc_txt.split('\n')    
    excel_y_begin = int(loc_arr[1])
    excel_x_begin = int(loc_arr[0])
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        img_name = img_file[:-4]
        print img_name
        #最原始的图
        img = cv2.imread(img_path)
        thresh_img = Pre_treat().gray_thresh_255(img)
        excel_img = Cut().excel(thresh_img, excel_y_begin, excel_x_begin)
        contain_img = Cut().contain(excel_img)
        #cv2.imwrite(img_name+"_contain_img.bmp", contain_img)
        cell_img_list = Cut().cell(contain_img)
        str_list = Cut().character(cell_img_list)
        text_list = Recognise().bin_to_text(str_list)
        result_list.extend(text_list)
    rem_list = remove_mu_a(result_list)
    df_flt_data = pd.DataFrame(result_list)
    str_time = time.strftime("%Y-%m-%d %H%M%S")
    output_path = u'C:/Users/Administrator/Desktop/航班数据采集/航班数据' + str_time + '.xlsx'
    df_flt_data.to_excel(output_path, encoding= 'utf-8', header=False, index=False)
    
    
                
        