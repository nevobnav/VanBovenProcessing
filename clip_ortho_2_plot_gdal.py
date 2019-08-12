# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:45:07 2019
@author: VanBoven
"""

import os
from vanbovendatabase.postgres_lib import *
import gdal
import time
from geoalchemy2.shape import to_shape
import geopandas as gpd
from fiona.crs import from_epsg
import shutil

config_file_path = r'/Users/Bas/Documents/config.json'
port = 5432

# con,meta = connect(DB_USER, DB_PASSWORD, DB_NAME, host=DB_IP)

#shape_file = '/Users/Bas/Desktop/test/test2.shp'

def clip_ortho2plot_gdal(this_plot_name, con, meta, ortho_ready_inbox, file):

    tic = time.time()
    
    input_file = os.path.join(ortho_ready_inbox,file)
    output_file = os.path.join(ortho_ready_inbox, str(file[:-4]) + '_clipped.VRT')
    
    # file and path names for temp shapefile
    shape_folder = os.path.join(ortho_ready_inbox, '000_temp_shapes')
    shape_path = os.path.join(shape_folder, 'tempshape.shp')
    
    if not(os.path.isdir(shape_folder)):
        os.makedirs(shape_folder)
    
    # get shape from database and store as physical shape file
    geometry = to_shape(get_plot_shape(this_plot_name, meta, con)).buffer(0.00006)
    input_shape = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs=from_epsg(4326))
    
    input_shape.to_file(shape_path, driver='ESRI Shapefile')
    
    # load orthomosaic with GDAL
    try:
        input_object = gdal.Open(input_file)
    except: 
        print('Could not load orthomosaic, check directory.')
        
    try:
        ds = gdal.Warp(output_file,
                       input_object,
                       format = 'VRT',
                       cutlineDSName = shape_path,
                       cutlineLayer = 'tempshape', 
                       warpOptions=['NUM_THREADS=ALL_CPUS'],
                       multithread=True,
                       warpMemoryLimit=3000,
                       transformerOptions=['NUM_THREADS=ALL_CPUS']
#                       dstAlpha= True,
#                       srcAlpha=True,
#                       dstNodata = 0
                       )
        if ds:
            toc = time.time()
            crop_time = toc-tic
            statement = 'Succesfully clipped {} to plot outline in {} seconds'
            print(statement.format(this_plot_name, crop_time))
            
            # clear GDAL object & remove temporary shapefile
            ds = None
            shutil.rmtree(shape_folder)
            
        else:
            statement = 'Clipping of {} failed - check used shape at ({}) and plot'
            print(statement.format(this_plot_name, shape_path))
        
    except: 
        statement = 'Clipping of {} failed - check used shape at ({}) and plot'
        print(statement.format(this_plot_name, shape_path))
        
        
