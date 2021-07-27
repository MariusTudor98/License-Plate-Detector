# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:22:14 2021

@author: Marius-Iulian TUDOR
"""
import cv2
import numpy as np
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras import layers, models
import keras
from PIL import Image


def preprocesare(image_path,resize=True): 
    img=cv2.imread(image_path) 
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img=img/255 
    if resize:
        img=cv2.resize(img, (480,480))
        
    return img

def areaSize(area, width, height):
    if (area < 1500 or area > 25000):
        return False
    return True

from datetime import datetime
start = datetime.now()
path=r'D:\licenta\set imagini autovehicule\1 (59).jpg'
x=cv2.imread(path)
x=cv2.resize(x,(480,480))
cv2.imshow('original', x)
file=preprocesare(path)
img_clean=preprocesare(path)

cv2.imshow('gray',file)
#filtru sobel
dx = ndimage.sobel(file, 1)  
dy = ndimage.sobel(file, 0) 
mag = np.hypot(dx, dy)       

cv2.imshow('sobel',mag)

mag = cv2.GaussianBlur(mag, (5, 5), 0)
cv2.imshow('gaussian',mag)
filename='sobel.png'
cv2.imwrite(filename, mag)
img=cv2.imread(r'D:\licenta\sobel.png', 0)
img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('global', th1)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu 1', th2)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu',th3)

img=th3
kernel1=np.ones((3,3), np.uint8)
erosion=cv2.erode(img,kernel1,iterations=1)
kernel=np.ones((7,7), np.uint8)
dilation=cv2.dilate(erosion,kernel,iterations=1)
cv2.imshow('Deschidere', dilation)
erosion=cv2.erode(dilation,kernel1,iterations=1)
cv2.imshow('Eroziune', erosion)
file=erosion

smooth = cv2.bilateralFilter(file, 9, 75, 75) 
cv2.imshow('bilateral', smooth)
edges=cv2.Canny(smooth, 100, 200) 
cv2.imshow('test', edges)

cnts,new = cv2.findContours(smooth.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
xy=x.copy()
cv2.drawContours(x, cnts, -1, (0,255,0),3)
cv2.imshow('Contur', x)

cnts0=cnts
cnts=sorted(cnts, key=cv2.contourArea, reverse= True)[:30] 
screenCnt= None #variabila unde stocam contulul nostru
cv2.drawContours(xy,cnts,-1,(0,255,0),3) 
cv2.imshow("sortare",xy) #top 30 contours


idx=0
for c in cnts:
        perimetru = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * perimetru, True)
        if len(approx) == 4: 
                screenCnt = approx
                x,y,w,h = cv2.boundingRect(c) 
                new_img=img_clean[y:y+h,x:x+w]
                min_rect=cv2.contourArea(c)
                if areaSize(min_rect,w,h):
                    
                   new_img=img_clean[y:y+h,x:x+w]
                   cv2.imwrite('./'+str(idx)+'.png',new_img) 
                   idx+=1
                   break

cv2.imshow("numar", new_img)
area=cv2.contourArea(c)
print (area)
nume='numar.png'
cv2.imwrite(nume, new_img)
kernel=np.ones((2,2), np.uint8)
new_img=cv2.dilate(new_img,kernel,iterations=1)
new_img=cv2.erode(new_img,kernel1,iterations=1)
cv2.imshow('deschidere', new_img)
new_img = cv2.normalize(src=new_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.threshold(new_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,new_img)
cv2.imshow('after_thresh', new_img)

contours2, image = cv2.findContours(new_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
c = max(contours2, key = cv2.contourArea)
x, y, width, height = cv2.boundingRect(c)
roi1 = new_img[y:y+height, x:x+width]

cv2.imshow("max contour", roi1)
contours2, image = cv2.findContours(roi1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

numar=roi1.copy()

contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:10]
sorted_ctrs = sorted(contours2, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )
cv2.imshow("contours", roi1)
cv2.waitKey(0)
list=[]
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = roi1[y:y+h, x:x+w]
    cv2.rectangle(numar,(x,y),(x+w, y+h),(99,255,0),2)

    list.append(roi)


classes = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z' }

for i in range(len(list)):
    print(list[i])

    cv2.imshow(str(i),list[i])
    cv2.imwrite(r'D:\licenta\clasificare\caractere{}.png'.format(str(i)), list[i])
    
model=keras.models.load_model(r'D:\licenta\model_char_recognition.h5')
width=28
height=28
channels=3
for i in range(1,len(list)):
        img=cv2.imread(r'D:\licenta\clasificare\caractere'+str(i)+'.png')
        img=Image.fromarray(img, "RGB")
        img=img.resize((height,width))
        img=np.array(img)
        img2=img.reshape((1,height,width,3))
        img2=img2/255.0
        prediction=model.predict(img2)
        char = np.argmax(prediction)
        print('Caracterul '+str(i)+' este ', classes[char])     
        
print(datetime.now() - start)

    






