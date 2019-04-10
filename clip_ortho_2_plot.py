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


def clip_ortho2plot(this_plot_name, con, meta, ortho_ready_inbox, file):
    with rasterio.open(os.path.join(ortho_ready_inbox, file)) as src:
        geometry = to_shape(get_plot_shape(this_plot_name, meta, con)).buffer(0.00006)
        geo = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs=from_epsg(4326))
        coords = getFeatures(geo)
        out_image, out_transform = rasterio.mask.mask(src, coords, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
    with rasterio.open(os.path.join(ortho_ready_inbox, str(file[:-4])+"_clipped.tif"), \
    "w", **out_meta, BIGTIFF='YES',NUM_THREADS='ALL_CPUS',COMPRESS='LZW') as dest:
        dest.write(out_image)

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]
