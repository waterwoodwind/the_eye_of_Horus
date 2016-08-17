# -*- coding: utf-8 -*-

import cv2
import os
import pandas as pd


from read_img import Load_Img
from pretreatment import Pre_Treatment
from layout_analysis import Header
from layout_analysis import Layout
from layout_analysis import Character
from recognition import Recognise
from filter import Filter


if __name__ == '__main__':
    list_flt_data = []
    img_list = Load_Img().get_img_path_list()
    for img_path in img_list:
        img_name = os.path.split(img_path)[1]
        print img_name
        img = cv2.imread(img_path)
        gray_img = Pre_Treatment().gray(img)

        header_start_index_list = Header().get_header_index(gray_img)
        print header_start_index_list
        # 字符在单元格中居左
        flt_number_start = header_start_index_list[3]
        plane_number_start = header_start_index_list[4]
        # 字符居中
        stand_number_start = header_start_index_list[5] + 5
        plan_arrive_start = header_start_index_list[18] + 5
        change_arrive_start = header_start_index_list[19] + 5

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
            str_plane_number = Recognise().np_to_digit(line, plane_number_start, plane_number_start + Character().width*6 + 1)
            str_stand_number = Recognise().np_to_digit(line, stand_number_start, stand_number_start + Character().width*3 + 1)
            str_plan_arrive = Recognise().np_to_digit(line, plan_arrive_start, plan_arrive_start + Character().width*4 + 1)
            str_change_arrive = Recognise().np_to_digit(line, change_arrive_start, change_arrive_start + Character().width*4 + 1)


            print str_flt_number,str_plane_number,str_stand_number,str_plan_arrive,str_change_arrive
            list_flt_data.append([str_flt_number,
                                  str_plane_number,
                                  str_stand_number,
                                  str_plan_arrive,
                                  str_change_arrive])

    no_repeat_list_flt_data = Filter().get_no_repeat(list_flt_data)
    own_company_flt_list = Filter().get_own_company(list_flt_data)
    no_repeat_last_list = Filter().get_no_repeat(own_company_flt_list)
    df_flt_data = pd.DataFrame(no_repeat_last_list)
    df_flt_data.to_csv(u'航班号_机号_机位.csv', encoding= 'utf-8', header=False, index=False)
    df_flt_data.to_excel(u'航班号_机号_机位.xlsx', encoding= 'utf-8', header=False, index=False)