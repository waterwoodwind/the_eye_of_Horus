# -*- coding: utf-8 -*-

import cv2
import os
from read_img import Load_Img
from pretreatment import Pre_Treatment
from layout_analysis import Header
from layout_analysis import Layout
from layout_analysis import Character
from recognition import Recognise

if __name__ == '__main__':
    img_list = Load_Img().get_img_path_list()
    for img_path in img_list:
        img_name = os.path.split(img_path)[1]
        print img_name
        img = cv2.imread(img_path)
        gray_img = Pre_Treatment().gray(img)

        header_start_index_list = Header().get_header_index(gray_img)
        print header_start_index_list
        flt_number_start = header_start_index_list[3]
        height_start_list = Layout().get_y_location(gray_img, header_start_index_list[0])
        print height_start_list
        thresh_img = Pre_Treatment().thres_inv(gray_img, 160)
        #对第一行蓝色选中数据进行反色
        ret,thresh_img[174-3:186+1+3,:] = cv2.threshold(thresh_img[174-3:186+1+3,:],254,255,cv2.THRESH_BINARY_INV)
        #取出文本块
        for index, starter in enumerate(height_start_list):
            line = thresh_img[starter-3:starter + 13 + 3, :]
            #cv2.imwrite('line/'+ img_name[:-4] + '_line_' + str(index) + '.bmp', line)
            str_flt_number = Recognise().np_to_digit(line, flt_number_start, flt_number_start + Character().width*6 + 1)


            print str_flt_number