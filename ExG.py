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

import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd

from sklearn.cluster import MeanShift

#import rasterio
import gdal
from osgeo import gdalnumeric

import rasterstats

os.chdir(r'C:\Users\VanBoven\Documents\GitHub\DataAnalysis')
import color_indices
import rasterstats_multicore

os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')
import clip_ortho_2_plot_gdal

def total_crop_area_per_gridcell(empty_grid, input_img_file, nodata_value):
    zs_list = []
    #count nr of valid pixels per grid cell
    zs = rasterstats.zonal_stats(empty_grid.geometry, input_img_file, nodata = nodata_value, 
                                 stats = 'mean')    
    [zs_list.append(count['mean']) for count in zs]
    # TODO to convert nr of pixels to area in metric units, first the gsd has to be determined
    
    #set values to grid attribute table
    grid = empty_grid.copy()
    grid['total_crop_area'] = zs_list      
    return grid

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
    #GLI_index = np.asarray(scaler.fit_transform(GLI_index), dtype=np.uint8)
    return GLI_index*255

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

def write_tif(img, raster, output_path, filename):
    with rasterio.open(raster) as src:
        #read
        out_transform = src.transform
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     'dtype': 'float64',
                     "count": 1,
                     "transform": out_transform})
    with rasterio.open(os.path.join(output_path, filename), \
    "w", **out_meta, BIGTIFF='YES',NUM_THREADS='ALL_CPUS',COMPRESS='LZW') as dest:
        dest.write(img)


raster = r"D:\VanBovenDrive\VanBoven MT\Archive\c03_termote\De Boomgaard\20190711\1144\Orthomosaic\c03_termote-De Boomgaard-201907111144-GR.tif"
clip_shp = r"D:\VanBovenDrive\VanBoven MT\Archive\c03_termote\De Boomgaard\20190514\2005\Clip_shape\clip_shape.shp"
#raster = r'E:\VanBovenDrive\VanBoven MT\Archive\c03_termote\Binnendijk Links\20190522\1625\Orthomosaic/c03_termote-Binnendijk Links-201905221625.tif'

output_path = r'D:\800 Operational\c03_termote\De Boomgaard\20190711\1144'
#srcArray = gdalnumeric.LoadFile(raster)
grid_path = r"D:\800 Operational\c03_termote\De Boomgaard\clip_shape_empty_grid.shp"


x_block_size = 5000 
y_block_size = 5000

#list to create subsest of blocks
it = list(range(0,250, 1))

# Function to read the raster as arrays for the chosen block size.
#def process_raster2template(x_block_size, y_block_size, model, skip, it):
tic = time.time()
i = 0


empty_grid = gpd.read_file(grid_path)
no_data_value = 255
zonal_stats = True
clip_ortho2shp = True

if clip_ortho2shp == True:
    ds = clip_ortho_2_plot_gdal.clip_ortho2shp_array(raster, clip_shp)
else:
    ds = gdal.Open(raster)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize
template = np.zeros([ysize, xsize], np.float32)
#define kernel for morhpological closing operation
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
            com1 = color_indices.BGR2COM1(img)
            template[y:y+rows, x:x+cols] = template[y:y+rows, x:x+cols] + com1

scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
template = np.asarray(scaler.fit_transform(template), dtype=np.uint8)
template.resize(1,template.shape[0],template.shape[1])

filename = os.path.basename(raster)[:-4] + '_com1.tif'
write_tif(template, raster, output_path, filename)

if zonal_stats == True:
    input_img_file = os.path.join(output_path,os.path.basename(raster)[:-4] + '_com1.tif')
    shp, tif = rasterstats_multicore.shp_tif(grid_path, raster)
    stats = rasterstats_multicore.run_zonalstats(shp, tif)    
    
    
    
    grid = total_crop_area_per_gridcell(empty_grid, input_img_file, no_data_value)
    grid.to_file(os.path.join(output_path, filename[:-4]+'_ZS.tif'))

            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
            #array = ds.ReadAsArray(x, y, cols, rows)
            #array = array[0:3,:,:]
            #if img.mean() > 0:
# =============================================================================
#             ExG2 = ExG(b,g,r)
#             cv2.imwrite(output_path + '/ExG_'+str(blocks)+'.jpg',ExG2)
#             ExG2 = ExG2.reshape(1, ExG2.shape[0], ExG2.shape[1]) 
#             #define filename
#             filename = 'ExG.tif'
#             write_tif(ExG2, raster, output_path, filename)
#             ExG2 = None
#             
#             var = VARI(b,g,r)
#             cv2.imwrite(output_path + '/VARI_'+str(blocks)+'.jpg',var)
#             var = var.reshape(1, var.shape[0], var.shape[1])
#             #define filename
#             filename = 'var.tif'
#             write_tif(var, raster, output_path, filename)
#             var = None
# =============================================================================
            
# =============================================================================
#             gli = GLI(b,g,r)
#             cv2.imwrite(output_path + '/GLI_'+str(blocks)+'.jpg',gli)
#             gli = gli.reshape(1, gli.shape[0], gli.shape[1])
#             filename = 'GLI2.tif'
#             write_tif(gli, raster, output_path, filename)
#             gli = None
#             
#             vndvi = visual_NDVI(b,g,r)
#             cv2.imwrite(output_path + '/vndvi_'+str(blocks)+'.jpg',vndvi)           
#             vndvi = vndvi.reshape(1, vndvi.shape[0], vndvi.shape[1])
#             filename = 'vndvi.tif'
#             write_tif(vndvi, raster, output_path, filename)
#             vndvi = None
#             
#             rgbvi = rgbvi(b,g,r)
#             cv2.imwrite(output_path + '/rgbvi_'+str(blocks)+'.jpg',rgbvi)
#             rgbvi = rgbvi.reshape(1, rgbvi.shape[0], rgbvi.shape[1])
#             write_tif(rgbvi, raster, output_path, filename)
#             rgbvi = None
#             
#             thresh, binary = cv2.threshold(ExG2, 0, 255, cv2.THRESH_OTSU)
#             cv2.imwrite(output_path + '\otsu_ExG.jpg', binary)
# 
#             thresh, binary = cv2.threshold(gli, 0, 255, cv2.THRESH_OTSU)
#             thresh2, binary = cv2.threshold(gli, thresh+10, 255, cv2.THRESH_BINARY)
#             cv2.imwrite(output_path + '\otsu_gli.jpg', binary)
# 
# cielab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# cv2.imwrite(output_path + '\cielab.jpg', cielab)
# 
# 
# =============================================================================
