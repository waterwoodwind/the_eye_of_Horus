# -*- coding: utf-8 -*-
import cv2

class Pre_Treatment(object):
    def __init__(self):
        pass

    def gray(self, img_np):
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        return img_gray

    def thres_inv(self, img_gray, split_number):
        ret, thresh_img = cv2.threshold(img_gray,split_number,255,cv2.THRESH_BINARY_INV)
        return thresh_img