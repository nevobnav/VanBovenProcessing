# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:49:27 2019

@author: VanBoven
"""


from sklearn import preprocessing
import pandas as pd
import os

import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift

#import rasterio
import gdal
from osgeo import gdalnumeric

def ExG(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    ExG_index = np.asarray((2.0*g - b - r), dtype=np.float32)
    #ExG_index[ExG_index < 0] = 0
    ExG_index = np.asarray(scaler.fit_transform(ExG_index),dtype=np.uint8)    
    return ExG_index

def VARI(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    VARI_index = np.where(g+r-b != 0, ((g-r)/(g+r-b)), ((g-r)/(g+r-b)).min())
    VARI_index[VARI_index > 1] = 0
    VARI_index[VARI_index <-1] = 0
    VARI_index = np.asarray(scaler.fit_transform(VARI_index), dtype=np.uint8)
    return VARI_index

def GLI(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    GLI_index = np.where(2*g+r+b != 0, ((2*g-r-b)/(2*g+r+b)), ((2*g-r-b)/(2*g+r+b)).min())
    GLI_index[GLI_index > 1] = 0
    GLI_index[GLI_index <-1] = 0
    GLI_index = np.asarray(scaler.fit_transform(GLI_index), dtype=np.uint8)
    return GLI_index

def visual_NDVI(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    visual_NDVI_index = np.where(g+r != 0, ((g-r)/(g+r)), ((g-r)/(g+r)).min())
    visual_NDVI_index[visual_NDVI_index > 1] = 0
    visual_NDVI_index[visual_NDVI_index <-1] = 0    
    visual_NDVI_index = np.asarray(scaler.fit_transform(visual_NDVI_index), dtype=np.uint8)
    return visual_NDVI_index    

def rgbvi(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    rgbvi_index = np.where((g*g) + (r*b) != 0, (((g*g) - (r*b))/((g*g)+(r*b))), (((g*g) - (r*b))/((g*g)+(r*b))).min())
    rgbvi_index[rgbvi_index > 1] = 0
    rgbvi_index[rgbvi_index < -1] = 0
    rgbvi_index = np.asarray(scaler.fit_transform(rgbvi_index), dtype = np.uint8)
    return rgbvi_index
    
x_block_size = 60000
y_block_size = 60000

#list to create subsest of blocks
it = list(range(0,10, 1))

# Function to read the raster as arrays for the chosen block size.
#def process_raster2template(x_block_size, y_block_size, model, skip, it):
tic = time.time()
i = 0
raster = r'E:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190513\1357\Orthomosaic/c08_biobrass-AZ74-201905131357_clipped.tif'
raster = r'E:\VanBovenDrive\VanBoven MT\Archive\c03_termote\Binnendijk Links\20190522\1625\Orthomosaic/c03_termote-Binnendijk Links-201905221625.tif'


output_path = r'F:\400 Data analysis\410 Plant count\c03_termote\Binnendijk_links'
#srcArray = gdalnumeric.LoadFile(raster)
ds = gdal.Open(raster)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize
template = np.zeros([ysize, xsize], np.uint8)
#define kernel for morhpological closing operation
kernel = np.ones((7,7), dtype='uint8')
blocks = 0
for y in range(0, ysize, y_block_size):
    if y + y_block_size < ysize:
        rows = y_block_size
    else:
        rows = ysize - y
    for x in range(0, xsize, x_block_size):
        blocks += 1
        #if statement for subset
        if blocks in it:
            if x + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x
            b = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
            g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
            r = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
            print('loaded img in memory')
            img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
            img[:,:,0] = b
            img[:,:,1] = g
            img[:,:,2] = r
            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
            #array = ds.ReadAsArray(x, y, cols, rows)
            #array = array[0:3,:,:]
            #if img.mean() > 0:
            ExG2 = ExG(b,g,r)
            cv2.imwrite(output_path + '/ExG_'+str(blocks)+'.jpg',ExG2)
            ExG2 = None
            var = VARI(b,g,r)
            cv2.imwrite(output_path + '/VARI_'+str(blocks)+'.jpg',var)
            var = None
            gli = GLI(b,g,r)
            cv2.imwrite(output_path + '/GLI_'+str(blocks)+'.jpg',gli)
            gli = None
            vndvi = visual_NDVI(b,g,r)
            cv2.imwrite(output_path + '/vndvi_'+str(blocks)+'.jpg',vndvi)
            vndvi = None
            rgbvi = rgbvi(b,g,r)
            cv2.imwrite(output_path + '/rgbvi_'+str(blocks)+'.jpg',rgbvi)
            rgbvi = None
            
            thresh, binary = cv2.threshold(ExG2, 0, 255, cv2.THRESH_OTSU)
            cv2.imwrite(output_path + '\otsu_ExG.jpg', binary)

            thresh, binary = cv2.threshold(gli, 0, 255, cv2.THRESH_OTSU)
            thresh2, binary = cv2.threshold(gli, thresh+10, 255, cv2.THRESH_BINARY)
            cv2.imwrite(output_path + '\otsu_gli.jpg', binary)

cielab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
cv2.imwrite(output_path + '\cielab.jpg', cielab)




#testing of meanshift
          
img = np.zeros((b.shape[0], b.shape[1], 3))
img[:,:,0] = b
img[:,:,1] = g
img[:,:,2] = r  
          
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
a = np.array(img_lab[:,:,1])
b2 = np.array(img_lab[:,:,2])

a_flat = a.flatten()
b2_flat = b2.flatten()
Classificatie_Lab = np.ma.column_stack((a_flat, b2_flat))

            
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)


