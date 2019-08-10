# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:45:07 2019
@author: VanBoven
"""

import os
from vanbovendatabase.postgres_lib import *
import json
import gdal
import multiprocessing
import gdal2tiles
import time
from geoalchemy2.shape import to_shape
import geopandas as gpd
from fiona.crs import from_epsg

config_file_path = r'/Users/Bas/Documents/config.json'
port = 5432

with open(config_file_path) as config_file:
    config = json.load(config_file)
DB_NAME = config['DB_NAME']
DB_USER = config['DB_USER']
DB_PASSWORD = config['DB_PASSWORD']
DB_IP = config['DB_IP']

# clip_ortho2plot(plot_name, con, meta, path_ready_to_upload,filename)

con,meta = connect(DB_USER, DB_PASSWORD, DB_NAME, host=DB_IP)

file = r'c01_verdonk-Wever west-201907170749-GR.tif'
ortho_ready_inbox = r'/Users/Bas/Desktop/test'
this_plot_name = 'Wever west'

#shape_file = '/Users/Bas/Desktop/test/test2.shp'

def clip_ortho2plot(this_plot_name, con, meta, ortho_ready_inbox, file):

    input_file = os.path.join(ortho_ready_inbox,file)
    output_file = os.path.join(ortho_ready_inbox, str(file[:-4]) + '_clipped.VRT')
    
    geometry = to_shape(get_plot_shape(this_plot_name, meta, con)).buffer(0.00006)
    input_shape = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs=from_epsg(4326))
    
    input_shape.to_file('/Users/Bas/Desktop/test/test2.shp', driver='ESRI Shapefile')
    
    
#    input_shape = get_plot_shape(this_plot_name, meta, con).buffer(0.00006)

    input_object = gdal.Open(input_file)
    
    try:
        ds = gdal.Warp(output_file,
                       input_object,
                       format = 'VRT',
                       cutlineDSName = shape_file,
                       cutlineLayer = 'test2'
#                       cutlineSQL = input_shape,
#                       dstAlpha= True,
#                       srcAlpha=True,
#                       dstNodata = 0
                       )
        ds = None
        
        
        
    except: 
        print('Cutting to shape did not work')
    
    
clip_ortho2plot(this_plot_name, con, meta, ortho_ready_inbox, file)




no_of_cpus = multiprocessing.cpu_count()
vrt_file = os.path.join(ortho_ready_inbox, str(file[:-4]) + '_clipped.VRT')

output_folder = r'/Users/Bas/Desktop/tiles'
zoomlevel = 18

tiling_options = {'zoom': (16, zoomlevel), 'tmscompatible': True, 'nb_processes':no_of_cpus}

tic = time.time()
gdal2tiles.generate_tiles(vrt_file, output_folder, **tiling_options)
toc = time.time()

print('total time is', toc-tic)
    
    

#    output_epsg=4326
#    dst_srs = osr.SpatialReference()
#    dst_srs.ImportFromEPSG(output_epsg)
#    dst_wkt = dst_srs.ExportToWkt()
#    output_Type = gdal.GDT_Byte
#
#    warpopts = gdal.WarpOptions(format='VRT',
#                                outputType=output_Type,
#                                workingType=output_Type,
#                                srcSRS=dst_wkt,
#                                dstSRS=dst_wkt,
#                                dstAlpha=True,
#                                warpOptions=['NUM_THREADS=ALL_CPUS'],
#                                warpMemoryLimit=3000,
#                                creationOptions=['COMPRESS=LZW','TILED=YES', 'NUM_THREADS=ALL_CPUS', 'SKIP_NOSOURCE=YES'],
#                                multithread=True,
#                                transformerOptions=['NUM_THREADS=ALL_CPUS'],
#                                cutlineDSName = shapyyy,
#                                cropToCutline=True,
#                                )

    # Perform actual warping operation -> output to specified path, filename
#    output_object = gdal.Warp(output_path, input_path, options = warpopts)


    # with rasterio.open(os.path.join(ortho_ready_inbox, file)) as src:
    #     geometry = to_shape(get_plot_shape(this_plot_name, meta, con)).buffer(0.00006)
    #     geo = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs=from_epsg(4326))
    #     coords = getFeatures(geo)
    #     out_image, out_transform = rasterio.mask.mask(src, coords, crop=True)
    #     out_meta = src.meta
    # out_meta.update({"driver": "GTiff",
    #                          "height": out_image.shape[1],
    #                          "width": out_image.shape[2],
    #                          "transform": out_transform})
    # with rasterio.open(os.path.join(ortho_ready_inbox, str(file[:-4])+"_clipped.tif"), \
    # "w", **out_meta, BIGTIFF='YES',NUM_THREADS='ALL_CPUS',COMPRESS='LZW') as dest:
    #     dest.write(out_image)

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]







#geo.to_file('dataframe.shp', driver='ESRI Shapefile')
#shapyyy = r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing/dataframe.shp'