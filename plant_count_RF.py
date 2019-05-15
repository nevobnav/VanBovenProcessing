# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:35:13 2019

@author: VanBoven
"""

import os
os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')

import pandas as pd

import time
import cv2
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans

from sklearn import preprocessing
import Random_Forest_Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_blobs

#import rasterio
import gdal
from osgeo import gdalnumeric

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]]);
            image[y:y + windowSize[1], x:x + windowSize[0]] = window

def ExG(b,g,r):
    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 255))
    ExG_index = np.asarray((2.0*g - b - r), dtype=np.float32)
    ExG_index[ExG_index < 0] = 0
    ExG_index = np.asarray(scaler2.fit_transform(ExG_index),dtype=np.uint8)    
    return ExG_index


train_data_path = r'E:\400 Data analysis\410 Plant count\Training_data'

#get trained RFC model
model, scaler = Random_Forest_Classifier.get_trained_model(train_data_path)

#read image

x_block_size = 4096
y_block_size = 4096

#list to create subsest of blocks
it = list(range(6,500, 2))
#skip = True if you do not want to process each block but you want to process the entire image
#skip = True
# Function to read the raster as arrays for the chosen block size.
#tic = time.time()
i = 0
raster = r"E:\VanBovenDrive\VanBoven MT\Archive\c00_development\Testdag - Liederik\Orthomosaic/20190319.tif"
#srcArray = gdalnumeric.LoadFile(raster)
ds = gdal.Open(raster)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

b2 = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
img2 = np.zeros([b2.shape[0],b2.shape[1],3], np.uint8)
img2[:,:,0] = b2
b2 = None
g2 = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
img2[:,:,1] = g2
g2 = None
r2 = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))
img2[:,:,2] = r2
r2 = None

#template = np.zeros([ysize, xsize], np.uint8)
#define kernel for morhpological closing operation
#kernel = np.ones((7,7), dtype='uint8')
blocks = 0
for y2 in range(0, ysize, y_block_size):
    if y2 + y_block_size < ysize:
        rows = y_block_size
    else:
        rows = ysize - y2
    for x2 in range(0, xsize, x_block_size):
        tic2 = time.time()
        blocks += 1
        #if statement for subset
        if blocks in it:
            if x2 + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x2
            b = np.array(ds.GetRasterBand(1).ReadAsArray(x2, y2, cols, rows)).astype(np.uint(8))
            g = np.array(ds.GetRasterBand(2).ReadAsArray(x2, y2, cols, rows)).astype(np.uint(8))
            r = np.array(ds.GetRasterBand(3).ReadAsArray(x2, y2, cols, rows)).astype(np.uint(8))
            img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
            img[:,:,0] = b
            img[:,:,1] = g
            img[:,:,2] = r
            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
            #array = ds.ReadAsArray(x, y, cols, rows)
            #array = array[0:3,:,:]
            if img.mean() > 0:
                stepSize = 250
                windowSize = (40, 40)
                for (x, y, window) in sliding_window(img, stepSize, windowSize):
                    tic = time.time()
                    if window.mean() > 0: 
                        if window.shape[0] > 5 and window.shape[1] > 5:
                            output_features = Random_Forest_Classifier.get_image_features(window, scaler)                                
                            prediction = str(model.predict(output_features)[0])
                            #print(prediction)
                            #i+=1
                            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling_RF/RF_window_' +str(i)+str(prediction)+'.jpg', window)
                            if prediction == "Broccoli":
                                i+=1
                                b,g,r = (window[:,:,0], window[:,:,1], window[:,:,2])
                                #calc excess_green index
                                ExG_index = ExG(b,g,r)
                                #apply otsus thresholding to get most prominent green features
                                thresh, hough_test = cv2.threshold(ExG_index, 0, 255, cv2.THRESH_OTSU)
                                #apply morphological closing to make plants solid objects
                                kernel = np.ones((9,9), dtype='uint8')
                                closing = cv2.morphologyEx(hough_test, cv2.MORPH_CLOSE, kernel)
                                contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                #template = np.zeros(img2.shape).astype(np.uint8)
                                for cnt in contours:
                                    ar = cv2.contourArea(cnt)
                                    if (ar > 9) & (ar < 1001):    
                                        M = cv2.moments(cnt)
                                        try:
                                            cx = int(M['m10']/M['m00'])
                                            cy = int(M['m01']/M['m00'])
                                        except:
                                            print('0')
                                        cv2.drawMarker(window, (cx,cy), (0,0,255), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 2)
                                        cv2.drawContours(window, cnt,-1, (255, 255, 255),-1)
                    toc = time.time()
                    #print('Processing of one window took '+str(toc-tic)+' seconds')
                img2[y2:y2+rows, x2:x2+cols] = img
                toc2 = time.time()
                print('Processing of one block took ' + str(toc2-tic2)+' seconds')
                print('Number of broccoli after this block is '+str(i))
                cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling_RF/RF_test.jpg', img2)
                
                


                    

def window_stack(a, stepsize=500, width=40):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width))                     
      
tic = time.time()              
test = window_stack(img2)
toc=time.time()


