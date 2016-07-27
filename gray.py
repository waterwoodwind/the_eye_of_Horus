# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:03:53 2016

@author: Administrator
"""

import cv2
import os
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
        data_img = gray[171:1006+1,4:1907+1]
        ret,thresh_img = cv2.threshold(data_img,208,255,cv2.THRESH_BINARY_INV)
        return thresh_img
    
    def y_x_border_list(self, img):
        y_s_list = self.y_shadow_list(img)
        y_start_list, y_end_list = self.shadow_border(y_s_list)
        x_s_list = self.x_shadow_list(img)
        x_start_list, x_end_list = self.shadow_border(x_s_list)
        return y_start_list, y_end_list, x_start_list, x_end_list         
        
img_dir = "single_img"
for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    print img_path
    thresh_img = Pre_treat().get_data_thresh_img(img_path)
    
    y_start_list, y_end_list, x_start_list, x_end_list = \
    Pre_treat().y_x_border_list(thresh_img)
    print y_start_list, y_end_list, x_start_list, x_end_list
    
    #有选中蓝色背景的去除第一个end
    y_end_list.pop(0)
    line_1 = thresh_img[y_start_list[0]:y_end_list[0]+1,:]
    cv2.imshow("line_1", line_1)
    cv2.waitKey(0)
    x_s_list = Pre_treat().x_shadow_list(line_1)
    x_start_list, x_end_list = Pre_treat().shadow_border(x_s_list)
    
    Character_1 = line_1[:, x_start_list[1]:x_end_list[1]+1]
    cv2.imshow("Character_1", Character_1)
    cv2.waitKey(0)
    #cv2.imwrite('Character_1.bmp', Character_1)