# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 20:01:57 2019

@author: VanBoven
"""

from osgeo import gdal




InputImage = r'C:\Users\VanBoven\Desktop\TEST\c01_verdonk-Wever oost-201907170731-GR BCK.tif'
Image = gdal.Open(InputImage, 1) # 0 = read-only, 1 = read-write.

gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetCacheMax = 2000

#translateoptions = gdal.SetConfigOption(gdal.ParseCommandLine("-co COMPRESS=LZW -co NUM_THREADS=ALL_CPUS"))


#opts = gdal.SetConfigOption(format='GTiff',
#                            warpOptions=['NUM_THREADS=ALL_CPUS'],
#                            warpMemoryLimit=3000,
#                            creationOptions=['COMPRESS=LZW','TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS', 'JPEG_QUALITY=100', 'BIGTIFF=YES', 'ALPHA=YES'],
#                            resampleAlg='cubicspline',
#                            multithread=True,
#                            tps=tps_flag,
#                            transformerOptions=['NUM_THREADS=ALL_CPUS']
#                            )



Image.BuildOverviews("NEAREST", [2,4,6,8])

#Image.BuildOverviews("NEAREST", [2,4,8,16,32,64])
del Image
print('Done.')