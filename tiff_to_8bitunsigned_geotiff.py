# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:22:03 2019

@author: VanBoven
"""


#import rasterio
import gdal
from osgeo import gdalnumeric
from osgeo import osr

import numpy as np

def getNoDataValue(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()

def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    #Specify here the number of bands, rows, cols and dtype of output
    outRaster = driver.Create(newRasterfn, cols, rows, 3, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    
    bands = array.shape[2]
    for band in range(bands):
        outRaster.GetRasterBand(band+1).WriteArray( array[:, :, band] )
        outRaster.GetRasterBand(band+1).SetNoDataValue(0)
    #outband = outRaster.GetRasterBand(1)
    #outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outRaster.FlushCache()

src = gdal.Open(r"E:\VanBovenDrive\VanBoven MT\Archive\c04_verdegaal\Frederikslaan\20190420\Orthomosaic/c04_verdegaal-Frederikslaan-20190420.tif")

b = src.GetRasterBand(1).ReadAsArray()#.astype(np.uint16)
b = np.asarray((b / b.max()) * 255, dtype = np.uint8)
g = src.GetRasterBand(2).ReadAsArray()#.astype(np.uint16)
g = np.asarray((g / g.max()) * 255, dtype = np.uint8)
r = src.GetRasterBand(3).ReadAsArray()#.astype(np.uint16)
r = np.asarray((r / r.max()) * 255, dtype = np.uint8)

img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
img[:,:,0] = b
img[:,:,1] = g
img[:,:,2] = r

rasterfn = r"E:\VanBovenDrive\VanBoven MT\Archive\c04_verdegaal\Frederikslaan\20190420\Orthomosaic/c04_verdegaal-Frederikslaan-20190420.tif"
newValue = 0
newRasterfn = r'C:\Users\VanBoven\Documents\100 Ortho Inbox/c04_verdegaal-Frederikslaan-20190420.tif'

# Convert Raster to array
#rasterArray = raster2array(rasterfn)
# Get no data value of array
noDataValue = getNoDataValue(rasterfn)
img[img < 0 ] = 0
# Updata no data value in array with new value
img[img == noDataValue] = newValue

# Write updated array to new raster
array2raster(rasterfn,newRasterfn,img)


