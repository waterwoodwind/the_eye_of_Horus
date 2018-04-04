# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 06:25:02 2018

@author: wangxin
"""

#encoding: utf-8
import os
import pygame
import cv2

chinese_dir = 'ascii'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()
start,end = (0x30,0x5b) # 大写字母编码范围
#33,126 ascii字符范围
for codepoint in range(33,126+1):
    word = unichr(codepoint)
    font = pygame.font.Font("simsun.ttc", 13)
    # 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
    # 64是生成汉字的字体大小
    rtext = font.render(word, False, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(chinese_dir, str(codepoint) + ".png"))
    img_dir = os.path.join(chinese_dir, str(codepoint) + ".png")
    img = cv2.imread(img_dir, 0)
    ret,thresh_img = cv2.threshold(img,160,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(img_dir, thresh_img)

#word = u'航'   
#rtext = font.render(word, False, (0, 0, 0), (255, 255, 255))
#pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))