#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:05:29 2019

@author: Bas
"""

from osgeo import gdal



input_file =r'/Users/Bas/Desktop/c01_verdonk-Wever west-201907240724.tif'

input_object = gdal.Open(input_file,1)

gdaloptions = {'COMPRESS_OVERVIEW': 'JPEG', 
               'PHOTOMETRIC_OVERVIEW': 'YCBR',
               'INTERLEAVE_OVERVIEW': 'PIXEL',
               'NUM_THREADS': 'ALL_CPUS'
               }

for key, val in gdaloptions.items():
    gdal.SetConfigOption(key,val)

gdal.SetCacheMax = 3000
input_object.BuildOverviews("NEAREST", [8,16,32,64,128])
input_object = None
