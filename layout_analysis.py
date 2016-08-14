# -*- coding: utf-8 -*-
import cv2

class Projection(object):

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

class Header(object):

    def __init__(self):
        self.height_start = 119
        self.height_end = 163
        self.width_start = 2
        self.width_end = 1889

    def get_header_index(self, gray):
        gray_header = gray[self.height_start:self.height_end, \
                           self.width_start:self.width_end]
        ret,thresh_header = cv2.threshold(gray_header,220,255,cv2.THRESH_BINARY_INV)
        x_shadow = Projection().x_shadow_list(thresh_header)
        start_index,end_index = Projection().shadow_border(x_shadow)
        start_index = map(lambda x:x+2, start_index)
        return start_index

    def plan_arrive_start(self, gray):
        start_index = self.header_index(gray)
        return start_index[18]

    def change_arrive_start(self, gray):
        start_index = self.header_index(gray)
        return start_index[19]

class Flt_data_zone(object):
    def __init__(self):
        self.height_start = 167
        self.height_end = 1002
        self.index_width = 44

class Character(object):
    def __init__(self):
        self.height = 13
        self.width = 9
        self.head = 3

class Layout(object):
    def __init__(self):
        pass

    def get_thresh_img_index(self, img, index_width_start):
        img_index = img[Flt_data_zone().height_start:Flt_data_zone().height_end, index_width_start:index_width_start + Flt_data_zone().index_width]
        ret,img_thresh_index = cv2.threshold(img_index,160,255,cv2.THRESH_BINARY_INV)
        return img_thresh_index

    def get_y_location(self, img, index_width_start):
        img_thresh_index = self.get_thresh_img_index(img, index_width_start)
        y_start_list, y_end_list, x_start_list, x_end_list = \
        Projection().y_x_border_list(img_thresh_index)
        y_start_list, y_end_list = \
        Projection().del_surplus_y_line(y_start_list, y_end_list)

        #加入第一行蓝色区域
        y_start_list.insert(0, 7)
        y_end_list.insert(0, 19)
        height_start_list = map(lambda x:x+Flt_data_zone().height_start, y_start_list)

        return height_start_list

