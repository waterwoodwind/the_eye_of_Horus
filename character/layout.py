#coding=utf-8
import os

import cv2

from character.gray import Pre_treat
from character.gray import clear_dir


class header(object):
    
    def __init__(self):
        self.height_start = 119
        self.height_end = 163
        self.width_start = 2
        self.width_end = 1889

    def header_index(self, gray):
        gray_header = gray[header().height_start:header().height_end, \
                           header().width_start:header().width_end]
        ret,thresh_header = cv2.threshold(gray_header,220,255,cv2.THRESH_BINARY_INV)
        x_shadow = Pre_treat().x_shadow_list(thresh_header)
        start_index,end_index = Pre_treat().shadow_border(x_shadow)
        return start_index
        
    def plan_arrive_start(self, gray):
        start_index = self.header_index(gray)
        return start_index[18] + 5
    
    def change_arrive_start(self, gray):
        start_index = self.header_index(gray)
        return start_index[19] + 5
        
if __name__ == '__main__':
    clear_dir()
    character_list = []
    #img_dir = "multi_img"
    img_dir = "single_img"
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        img_name = img_file[:-4]
        print img_path
        #最原始的图
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_header = gray[header().height_start:header().height_end, \
                           header().width_start:header().width_end]
        ret,thresh_header = cv2.threshold(gray_header,220,255,cv2.THRESH_BINARY_INV)
        x_shadow = Pre_treat().x_shadow_list(thresh_header)
        start_index,end_index = Pre_treat().shadow_border(x_shadow)
        plan_arrive_start = start_index[18]
        change_arrive_start = start_index[19]