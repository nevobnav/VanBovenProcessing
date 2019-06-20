# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:25:53 2019

@author: VanBoven
"""

import os
import rasterio.mask
import json
import geopandas as gpd


shp_path = r'F:\700 Georeferencing\AZ74 georeferencing/plot_extent.shp'
ortho_path = r'F:\700 Georeferencing\AZ74 georeferencing/' 
output_path = r'F:\700 Georeferencing\AZ74 georeferencing\clipped_imagery'


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def clip_ortho2plot(ortho_path, filename, shp_path, output_path):
    with rasterio.open(os.path.join(ortho_path, filename)) as src:
        #read shapefile
        gdf = gpd.read_file(shp_path)
        coords = getFeatures(gdf)
        out_image, out_transform = rasterio.mask.mask(src, coords, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
    with rasterio.open(os.path.join(output_path, filename), \
    "w", **out_meta, BIGTIFF='YES',NUM_THREADS='ALL_CPUS',COMPRESS='LZW') as dest:
        dest.write(out_image)

for filename in os.listdir(ortho_path):
    if filename.endswith('.tif'):
        clip_ortho2plot(ortho_path, filename, shp_path, output_path)
        