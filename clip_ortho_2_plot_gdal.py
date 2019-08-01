# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:45:07 2019

@author: VanBoven
"""

import os
from vanbovendatabase.postgres_lib import *
import rasterio.mask
import shapely
import numpy as np
import json
from geoalchemy2.shape import to_shape
import geopandas as gpd
from fiona.crs import from_epsg
import gdal

config_file_path = r'C:\Users\VanBoven\MijnVanBoven\config.json'
port = 5432


    clip_ortho2plot(plot_name, con, meta, path_ready_to_upload,filename)

def clip_ortho2plot(this_plot_name, con, meta, ortho_ready_inbox, file):

    input_path = os.path.join(ortho_ready_inbox,file)
    output_path = os.path.join(ortho_ready_inbox, str(file[:-4]) + '_clipped.tif')
    
    geometry = to_shape(get_plot_shape(this_plot_name, meta, con)).buffer(0.00006)
    geo = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs=from_epsg(4326))

    output_Type = gdal.GDT_Byte
    
    geo.to_file('dataframe.shp', driver='ESRI Shapefile')
    
    shapyyy = r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing/dataframe.shp'
    
    output_epsg=4326
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(output_epsg)
    dst_wkt = dst_srs.ExportToWkt()

    warpopts = gdal.WarpOptions(format='GTiff',
                                outputType=output_Type,
                                workingType=output_Type,
                                srcSRS=dst_wkt,
                                dstSRS=dst_wkt,
                                dstAlpha=True,
                                warpOptions=['NUM_THREADS=ALL_CPUS'],
                                warpMemoryLimit=3000,
                                creationOptions=['COMPRESS=LZW','TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=ALL_CPUS', 'JPEG_QUALITY=100', 'BIGTIFF=YES', 'SKIP_NOSOURCE=YES', 'ALPHA=YES'],
#                                resampleAlg='cubicspline',
                                multithread=True,
                                transformerOptions=['NUM_THREADS=ALL_CPUS'],
                                cutlineDSName = shapyyy,
                                cropToCutline=True,
                                )

    # Perform actual warping operation -> output to specified path, filename
    output_object = gdal.Warp(output_path, input_path, options = warpopts)




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
