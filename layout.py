#coding=utf-8
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Load an color image in grayscale
import cv2.cv as cv
image = cv.LoadImage('2.bmp')


new = cv.CreateImage(cv.GetSize(image), image.depth, 1)
for i in range(image.height):
    for j in range(image.width):
        new[i,j] = image[i,j][0] + image[i,j][1] + image[i,j][2]

cv.Threshold(new, new, 0, 255, cv.CV_THRESH_BINARY_INV)
cv.ShowImage('a_window', new)
cv.WaitKey(0)

line = [0 for x in range(image.height)]
for i in range(image.height):
    for j in range(image.width):
        if(new[i,j] == 255):        
            line[i] = line[i] + 1

for i in range(len(line)):
    if line[i] == 1920:
        line[i] = 0
        
print line
line.reverse()
x = np.arange(len(line))
plt.barh(x,line)
plt.show()