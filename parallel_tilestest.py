#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:52:15 2019

@author: Bas
"""

from osgeo import gdal
import gdal2tiles
import multiprocessing
import time

no_of_cpus = multiprocessing.cpu_count()

input_file =r'/Users/Bas/Desktop/vanboven/gdal_test/c08_biobrass-C49-201906041643_sample.tif'
output_file = r'/Users/Bas/Desktop/vanboven/gdal_test/c08_biobrass-C49-201906041643_sample_crop_VRT.VRT'

input_shape = r'/Users/Bas/Desktop/vanboven/gdal_test/testcrop.shp'


input_object = gdal.Open(input_file)

ds = gdal.Warp(output_file,
              input_object,
              format = 'VRT',
              cutlineDSName = input_shape,
              cutlineLayer = 'testcrop',
              dstNodata = 0
              )


output_folder = r'/Users/Bas/Desktop/tiles'
zoomlevel = 23

tiling_options = {'zoom': (16, zoomlevel), 'tmscompatible': True, 'nb_processes':no_of_cpus}

tic = time.time()
gdal2tiles.generate_tiles(input_file, output_folder, **tiling_options)
toc = time.time()

print('total time is', toc-tic)


warpopts = gdal.WarpOptions(format='VRT',
                            outputType=output_Type,
                            workingType=output_Type,
                            srcSRS=dst_srs,
                            dstSRS=dst_srs,
                            dstAlpha=True,
                            warpOptions=['NUM_THREADS=ALL_CPUS'],
                            warpMemoryLimit=3000,
                            creationOptions=['COMPRESS=LZW','TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS'],
                            resampleAlg='cubicspline',
                            multithread=True,
                            tps=tps_flag,
                            transformerOptions=['NUM_THREADS=ALL_CPUS'],
                            cutlineDSName = input_shape,
                            cutlineLayer = 'testcrop',
                            dstNodata = 0
                            )

input_raster = "path/to/rgb.tif"
output_raster = "path/to/rgb_output_cut.tif"
input_kml = "path/to/extent.kml"


ds = None