# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:19:35 2021

@author: divyam
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import math
import scipy.fftpack as fft

#cap = cv.VideoCapture('Ball_travel_10fps.mp4')
cap = cv.VideoCapture('Tag1.mp4')
#filt = cv.imread('ar_tag1.png')
#filt = cv.cvtColor(filt,cv.COLOR_BGR2GRAY)
#print(filt.shape)


if cap.isOpened() == False:
    print("Error opening the image")


count = 0 
img = None

while cap.isOpened():
    count += 1
    ret, frame = cap.read()
    if ret == False:
        break
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except:
        break
    gray = imutils.resize(gray,width = 400)
    _,thresh = cv.threshold(gray,200,255, cv.THRESH_BINARY_INV)

    if count == 50:
        #img = frame
        img = thresh
    
    cv.imshow('Frame',gray)
        
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()

print(img.shape)


img = np.float32(img)
#template = np.float32(template)
A_ = fft.fft2(img)
shift_A = fft.fftshift(A_)

A_mag = 20*np.log(np.abs(shift_A)+1)


## mask ##
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.ones((rows, cols), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0
 
res = shift_A*mask

C_shift = fft.ifftshift(res)
C = fft.ifft2(res)
C = np.abs(C)
plt.imshow(C)

#while(cv.waitKey(0) != 27):
#    cv.imshow('frame',img)
 

   
