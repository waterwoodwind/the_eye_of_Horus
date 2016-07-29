# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:43:49 2016

@author: Administrator
"""
import cv2
import os
import numpy as np
from gray import Pre_treat
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

dir_path = 'train'
test_dir_path = 'test'
def load_data(dir_path):
    list_img = []
    list_digits_target = []
    for digit_dir in os.listdir(dir_path):
        digit_path = os.path.join(dir_path, digit_dir)
        for img in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img)
            
            img = cv2.imread(img_path)
            print digit_dir, img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_one = gray.reshape(-1)
            list_img.append(gray_one)
            list_digits_target.append(int(digit_dir))
    
        
    np_all = np.array(list_img)
    np_digits = np.array(list_digits_target)
    return np_all, np_digits

np_all, np_digits = load_data(dir_path)
test_data, test_digits = load_data(test_dir_path)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(np_all, np_digits)

# Now predict the value of the digit on the second half:
expected = test_digits
predicted = classifier.predict(test_data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))