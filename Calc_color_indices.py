# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:49:27 2019

@author: VanBoven
"""


import pandas as pd
import os

import time
import cv2
import numpy as np

import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd

from joblib import Parallel, delayed
import itertools
import multiprocessing

from rasterstats import zonal_stats
import fiona

#import rasterio
import gdal
from osgeo import gdalnumeric

#import rasterstats

os.chdir(r'C:\Users\VanBoven\Documents\GitHub\DataAnalysis')
from raster_operations import reproject_raster_and_resample
import color_indices
from rasterstats_multicore import *

os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')
import clip_ortho_2_plot_gdal

def write_tif(img, raster, output_path, filename):
    with rasterio.open(raster) as src:
        #read
        out_transform = src.transform
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     'dtype': rasterio.float32,
                     "count": 1,
                     "transform": out_transform})
    with rasterio.open(os.path.join(output_path, filename), \
    "w", **out_meta, BIGTIFF='YES',NUM_THREADS='ALL_CPUS',COMPRESS='LZW') as dest:
        dest.write(img)

def chunks(data, n):
    """Yield successive n-sized chunks from a slice-able iterable."""
    for i in range(0, len(data), n):
        yield data[i:i+n]


def zonal_stats_partial(feats):
    """Wrapper for zonal stats, takes a list of features"""
    return zonal_stats(feats, tif, all_touched=True)

def total_crop_area_per_gridcell(empty_grid, input_img_file, nodata_value):
    zs_list = []
    #count nr of valid pixels per grid cell
    zs = zonal_stats(empty_grid.geometry, input_img_file, nodata = nodata_value, 
                                 stats = 'mean')    
    [zs_list.append(count['mean']) for count in zs]
    # TODO to convert nr of pixels to area in metric units, first the gsd has to be determined
    
    #set values to grid attribute table
    grid = empty_grid.copy()
    grid['total_crop_area'] = zs_list      
    return grid




#rasterpath = r'D:\700 Georeferencing\Hendrik de Heer georeferencing\clipped_imagery'
#rastername = ['20190503.tif', '20190513.tif', '20190522.tif', '20190527.tif', '20190603.tif']

rastername = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190619\1208\Orthomosaic\c07_hollandbean-Joke Visser-201906191208-GR.tif"]

# =============================================================================
# rastername = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190806\1119\Orthomosaic\c07_hollandbean-Hein de Schutter-201908061119-GR.tif",
#               r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190724\0955\Orthomosaic\c07_hollandbean-Hein de Schutter-201907240955-GR.tif",
#               r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190708\1101\Orthomosaic\c07_hollandbean-Hein de Schutter-201907081101-GR.tif",
#               r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190703\1301\Orthomosaic\c07_hollandbean-Hein de Schutter-201907031301-GR.tif",
#               r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190624\1432\Orthomosaic\c07_hollandbean-Hein de Schutter-201906241432-GR.tif",
#               r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190617\1419\Orthomosaic\c07_hollandbean-Hein de Schutter-201906171419-GR.tif",
#               r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190605\1255\Orthomosaic\c07_hollandbean-Hein de Schutter-201906051255-GR.tif"]
# 
# =============================================================================

for raster in rastername:
    #orthomosaic to process
    #raster = r"D:\700 Georeferencing\Hendrik de Heer georeferencing\clipped_imagery\20190522.tif"
    #raster = os.path.join(rasterpath, raster)
    #specify clip_shp if clip_ortho2shp is true
    clip_ortho2shp = True
    #optional clip_shape
    clip_shp = r"D:\800 Operational\c07_hollandbean\Joke visser\Clip_extent.shp"
    #output_path
    output_path = r'D:\800 Operational\c07_hollandbean\Joke visser\GLI'
    #optional path to empty grid to create geojson
    grid_path = r"D:\800 Operational\c07_hollandbean\Joke visser\Clip_extent_empty_grid.shp"
    
    #True if  you want to fill the grid with zonal stats of the calculated VI
    zonalstats = True
    #set no data value if zonal stats is True
    no_data_value = 255
    
    temp_tif_path = r'D:\800 Operational\c07_hollandbean\Hendrik de Heer\temp/temp2.tif'
    
    #block size for iterative processing 
    x_block_size = 5000 
    y_block_size = 5000
    
    #list to create subset of blocks
    it = list(range(0,250, 1))
    
    # Function to read the raster as arrays for the chosen block size.
    
    empty_grid = gpd.read_file(grid_path)
    
    if clip_ortho2shp == True:
        ds = clip_ortho_2_plot_gdal.clip_ortho2shp_array(raster, clip_shp)
    else:
        ds = gdal.Open(raster)
    ds_list = [ds]
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    template = np.zeros([ysize, xsize], np.float32)
    #define kernel for morhpological closing operation
    blocks = 0
    tic2 = time.time()
    for y in range(0, ysize, y_block_size):
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            tic = time.time()
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
                ind = color_indices.BGR2CieLAB(img)
                template[y:y+rows, x:x+cols] = template[y:y+rows, x:x+cols] + ind
                toc = time.time()
                print('Processing of block ' + str(blocks) + ' took '+str(toc-tic) + ' seconds')
    toc = time.time()
    print('Processing of full image took '+str(toc-tic2) + ' seconds')
    
    tic = time.time()
    #scale and resize output            
    #omin = template.min(axis=(0, 1), keepdims=True)
    #omax = template.max(axis=(0, 1), keepdims=True)
    #omin = -2
    #omax = 4
    omin = 0
    omax = 1
    
    try:
        template = np.array(((template - omin)/(omax - omin)))
    except:
        temp1 = template[:int(template.shape[0]/2), :int(template.shape[1]/2)]
        temp2 = template[int(template.shape[0]/2):, int(template.shape[1]/2):]
        temp1 = np.array(((temp1 - omin)/(omax - omin)))
        temp2 = template = np.array(((temp2 - omin)/(omax - omin)))        
        template[:int(template.shape[0]/2), :int(template.shape[1]/2)] = temp1
        template[int(template.shape[0]/2):, int(template.shape[1]/2):] = temp2
        
    template.resize(1,template.shape[0],template.shape[1])
    toc = time.time()
    print('scaling and resizing took '+str(toc-tic) + ' seconds')
    
    
    #write output
    filename = os.path.basename(raster)[:-4] + '_A.tif'
    write_tif(template, raster, output_path, filename)
    
    tic = time.time()
    reproject_raster_and_resample(input_path = os.path.join(output_path, filename), output_path = temp_tif_path, scaling_factor=400, destination_crs='EPSG:4326')
#    reproject_raster_and_resample(input_path = r"D:\800 Operational\c07_hollandbean\Hendrik de Heer\20190513\1422\20190513_ExG.tif", output_path = temp_tif_path, scaling_factor=400, destination_crs='EPSG:4326')
    toc = time.time()
    print('Resampling took ' + str(toc-tic) + ' seconds')
    
    tic = time.time()
    print('Started with calculating zonal stats...')
    if zonalstats == True:
        input_img_file = temp_tif_path
        #input_img_file = os.path.join(output_path,os.path.basename(raster)[:-4] + '_ExG.tif')
      
        grid = total_crop_area_per_gridcell(empty_grid, input_img_file, no_data_value)
        toc = time.time()
        print('Finished calculating zonal stats in ' + str(toc-tic) + ' seconds')
        grid.to_file(os.path.join(output_path, os.path.basename(raster)[:-4] + '_A.shp'))
    
# =============================================================================
#     shp = grid_path
#    # tif = temp_tif_path #os.path.join(output_path, filename)
#     
#     with fiona.open(shp) as src:
#         features = list(src)
# 
#     # Create a process pool using all cores
#     cores = multiprocessing.cpu_count()
#     p = multiprocessing.Pool(cores)
# 
#     # parallel map
#     stats_lists = p.map(zonal_stats_partial, chunks(features, cores))
# 
#     # flatten to a single list
#     stats = list(itertools.chain(*stats_lists))
# 
#     assert len(stats) == len(features)
#     print('Finished calculating zonal stats in ' + str(toc-tic) + ' seconds')
# =============================================================================



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
