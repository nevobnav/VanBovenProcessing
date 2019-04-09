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

with open(config_file_path) as config_file: 
    config = json.load(config_file)
    
host=config.get("DB_IP")
db=config.get("DB_NAME")
user=config.get("DB_USER")
password=config.get("DB_PASSWORD")

meta, con = connect(user, password, db, host, port=5432)

ortho_ready_inbox = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\ready'

for file in os.listdir(ortho_ready_inbox):
    if file.endswith('.tif'):
            this_customer_name,this_plot_name,this_date = file.split('-')
            this_date = this_date.split('.')[0]
            clip_ortho2plot(this_plot_name, con, meta, ortho_ready_inbox, file)

def clip_ortho2plot(this_plot_name, con, meta, ortho_ready_inbox, file):
    with rasterio.open(os.path.join(ortho_ready_inbox, file)) as src:
        geometry = to_shape(get_plot_shape(this_plot_name, con, meta))   
        geo = gpd.GeoDataFrame({'geometry': geometry}, index=[0], crs=from_epsg(4326))
        coords = getFeatures(geo)
        out_image, out_transform = rasterio.mask.mask(src, coords, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})            
    with rasterio.open(os.path.join(ortho_ready_inbox, str(file[:-4])+"_clipped.tif"), "w", **out_meta, BIGTIFF='YES') as dest:#, compress="JPEG"
        dest.write(out_image)
        
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]
