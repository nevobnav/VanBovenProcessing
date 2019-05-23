# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:49:27 2019

@author: VanBoven
"""


from sklearn import preprocessing
import pandas as pd

import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import rasterio
import gdal
from osgeo import gdalnumeric



def ExG(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    ExG_index = np.asarray((2.0*g - b - r), dtype=np.float32)
    ExG_index[ExG_index < 0] = 0
    ExG_index = np.asarray(scaler.fit_transform(ExG_index),dtype=np.uint8)    
    return ExG_index


x_block_size = 20000
y_block_size = 20000

#list to create subsest of blocks
it = list(range(0,20, 1))
#skip = True if you do not want to process each block but you want to process the entire image
skip = True
# Function to read the raster as arrays for the chosen block size.
#def process_raster2template(x_block_size, y_block_size, model, skip, it):
tic = time.time()
i = 0
raster = r"C:\Users\VanBoven\Desktop\20190514_termote_binnendijk_links_10m/20190514_c03_termote_binnendijklinks_test10m.tif"

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
            b = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
            g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
            r = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
            #img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
            #img[:,:,0] = b
            #img[:,:,1] = g
            #img[:,:,2] = r
            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
            #array = ds.ReadAsArray(x, y, cols, rows)
            #array = array[0:3,:,:]
            #if img.mean() > 0:
            ExG = ExG(b,g,r)
            cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c03_termote/10_m_binnendijk_test_'+str(blocks)+'.jpg',ExG)
                