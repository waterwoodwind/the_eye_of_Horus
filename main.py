# -*- coding: utf-8 -*-

import cv2
from read_img import Load_Img
from pretreatment import Pre_Treatment
from layout_analysis import Header
from layout_analysis import Layout

if __name__ == '__main__':
    img_list = Load_Img().get_img_path_list()
    for img_path in img_list:
        print img_path
        img = cv2.imread(img_path)
        gray_img = Pre_Treatment().gray(img)
        thresh_img = Pre_Treatment().thres_inv(gray_img, 160)
        header_start_index_list = Header().get_header_index(gray_img)
        #print header_start_index_list
        height_start_list = Layout().get_y_location(gray_img, header_start_index_list[0])
        print height_start_list

