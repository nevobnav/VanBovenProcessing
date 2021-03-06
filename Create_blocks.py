# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:56:21 2019

@author: VanBoven
"""


import pandas as pd

import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import rasterio
import gdal
from osgeo import gdalnumeric


x_block_size = 800
y_block_size = 800

#list to create subsest of blocks
it = list(range(0,7500, 5))
#skip = True if you do not want to process each block but you want to process the entire image
skip = True
# Function to read the raster as arrays for the chosen block size.
#def process_raster2template(x_block_size, y_block_size, model, skip, it):
tic = time.time()
i = 0
raster = r"E:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190513\1357\Orthomosaic/c08_biobrass-AZ74-201905131357_clipped.tif"
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
            img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
            img[:,:,0] = b
            img[:,:,1] = g
            img[:,:,2] = r
            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
            #array = ds.ReadAsArray(x, y, cols, rows)
            #array = array[0:3,:,:]
            if img.mean() > 0:
                cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c08_biobrass\AZ74/AZ74_'+str(blocks)+'.jpg',img)
                