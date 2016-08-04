# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 21:48:58 2016

@author: Administrator
"""
import os
import cv2
#相似度计算,inA、inB都是行向量
import numpy as np
from numpy import linalg as la
from sklearn.cluster import KMeans

#欧式距离
def euclidSimilar(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))
#皮尔逊相关系数
def pearsonSimilar(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]
#余弦相似度
def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=float(inA*inB.T)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def similar(inA, inB):
    print 'euclidSimilar:{eu}\npearsonSimilar:{pe}\ncosSimilar:{cos}'.format(eu=euclidSimilar(inA,inB), \
    pe=pearsonSimilar(inA,inB), \
    cos = cosSimilar(inA,inB))
    
if __name__ == '__main__':
    
    digit_path = 'no_repeat_character'
    list_img = []
    for img in os.listdir(digit_path):
        img_path = os.path.join(digit_path, img)
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_one = gray.reshape(-1)
        list_img.append(list(gray_one))
        
    np_all = np.array(list_img)
    
    random_state = 170
    y_pred = KMeans(init='k-means++', n_clusters=10, n_init=10)
    y_pred.fit(np_all)
    print y_pred