# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:45:07 2019
@author: VanBoven
"""

import os
os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')
from vanbovendatabase.postgres_lib import *
from osgeo import gdal
import time
from geoalchemy2.shape import to_shape
from osgeo import ogr, osr
#import geopandas as gpd
#from fiona.crs import from_epsg
import shutil

config_file_path = r'C:\Users\VanBoven\MijnVanBoven\config.json'
port = 5432

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
    #input_shape = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs={'init' :'epsg:4326'})
    
    driver = ogr.GetDriverByName("Esri Shapefile")
    ds = driver.CreateDataSource(shape_path)
    dest_srs = osr.SpatialReference()
    dest_srs.ImportFromEPSG(4326)
    layr1 = ds.CreateLayer('',dest_srs, ogr.wkbPolygon)
    layr1.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))    
    defn = layr1.GetLayerDefn()    
    feat = ogr.Feature(defn)
    feat.SetField('id', 1)  
    geom = ogr.CreateGeometryFromWkb(geometry.to_wkb())
    feat.SetGeometry(geom)
    layr1.CreateFeature(feat)
    ds.Destroy()
    #input_shape.to_file(shape_path, driver='ESRI Shapefile')
    
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
            statement = 'Succesfully clipped {} to plot outline in {:.2f} seconds'
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


def clip_ortho2shp_array(input_file, clip_shp):

    tic = time.time()
    output_file = ''
    # file and path names for temp shapefile
    shape_path = clip_shp
    shape_name = os.path.basename(clip_shp)[:-4]
    
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
                       cutlineLayer = shape_name,
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
            print(statement.format(input_file, crop_time))
            
        else:
            statement = 'Clipping of {} failed - check used shape at ({}) and plot'
            print(statement.format(input_file, shape_path))
        
    except: 
        statement = 'Clipping of {} failed - check used shape at ({}) and plot'
        print(statement.format(input_file, shape_path))
        
    return ds

        
        
