#coding=utf-8
import cv2
import os

class Load_Img(object):
    def __init__(self):
        pass

    def get_img_path_list(self, img_dir = "multi_img"):
        img_path_list = []
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            img_path_list.append(img_path)

        return img_path_list

    def read_img_list(self):
        img_list = []
        for img_path in self.get_img_path_list():
            img = cv2.imread(img_path)
            img_list.append(img)
        return img_list

if __name__ =='__main__':
    print Load_Img().get_img_path_list()

