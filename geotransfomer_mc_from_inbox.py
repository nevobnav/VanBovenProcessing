
## Translate input geotiff + gcps -> output tiled geotiff in correct coordinate system
## optimised for multi-core, lots of RAM computer

# INPUT:
# (1) GeoTiff in SPS EPSG:4326
# (2) .points file generated with Georeferencer in QGIS, containing GCPs in format [mapX, mapY, destX, destY]  in EPSG:4326

# OUTPUT:
# (1) GeoTiff file georeferenced according to defined GCPs


import csv
from osgeo import gdal, ogr, osr
import time
import sys
import datetime
import os

def coords_to_pixels(input_object, x, y):
    # The information is returned as tuple:
    # (TL x, X resolution, X rotation, TL y, Y rotation, y resolution)
    TL_x, x_res, _, TL_y, _, y_res = input_object.GetGeoTransform()

    # Divide the difference between the x value of the point and origin,
    # and divide this by the resolution to get the raster index
    x_index = (x - TL_x) / x_res

    # and the same as the y
    y_index = (y - TL_y) / y_res

    return x_index, y_index

def translate_and_warp_tiff(input_file, gcp_file, output_file):

    # read tiff with gdal
    input_object = gdal.Open(input_file)

    # open gcp points file and store in gcp_points
    with open(gcp_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader)
        gcp_list = []

        try:
            for row in reader:

                # construct gcp points string suitable for GDAL
                GCP_Pixel, GCP_Line = coords_to_pixels(input_object, float(row[2]), float(row[3]))

                #gdal.GCP(mapX, mapY, mapZ, sourcePixel, sourceLine)
                points = gdal.GCP(float(row[0]), float(row[1]), 0, GCP_Pixel, GCP_Line)
                gcp_list.append(points)
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(gcp_file, reader.line_num, e))

    # Set translation options for GDAL - hardcode reference system
    output_epsg=4326
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(output_epsg)
    dst_wkt = dst_srs.ExportToWkt()

    trnsopts = gdal.TranslateOptions(format='VRT',
                                     outputType=gdal.GDT_Byte,
                                     outputSRS=dst_wkt,
                                     GCPs=gcp_list,
                                     creationOptions=['NUM_THREADS = ALL_CPUS','TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
                                     )

    # Perform translate operation with GDAL -> output is VRT stored in memory
    translate_object = gdal.Translate('', input_object, options = trnsopts)

    # check if output is generated
    if translate_object is None:
        print('failed to perform translate operation')

    # remote input file & object from working memory
    input_object = None
    gdal.Unlink(input_file)

    # Set warping options for GDAL
    warpopts = gdal.WarpOptions(format='GTiff',
                                outputType=gdal.GDT_Byte,
                                workingType=gdal.GDT_Byte,
                                srcSRS=dst_wkt,
                                dstSRS=dst_wkt,
                                dstAlpha=True,
                                warpOptions=['NUM_THREADS=ALL_CPUS'],
                                warpMemoryLimit=3000,
                                creationOptions=['COMPRESS=LZW','TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=ALL_CPUS', 'JPEG_QUALITY=100', 'BIGTIFF=YES', 'SKIP_NOSOURCE=YES', 'ALPHA=YES'],
                                resampleAlg='cubicspline',
                                multithread=True,
                                tps=True,
                                transformerOptions=['NUM_THREADS=ALL_CPUS']
                                )

    # Perform actual warping operation -> output to specified path, filename
    output_object = gdal.Warp(output_file, translate_object, format='GTiff', options = warpopts)

    # check if output is generated
    if output_object is None:
        print('failed to perform warping operation')

    # clear memmory at end of script
    translate_object = None
    output_object = None

    toc = time.time()
    total_time = (toc-tic)

    return total_time

## CONFIG SECTION ##
inbox = r'C:\Users\VanBoven\Documents\100 Ortho Inbox' #folder where all orthos are stored

tic = time.time()

 #initialize
files = []
tif_count = 0
duplicate = 0
timestr = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
datestr = datetime.datetime.now().strftime('%Y%m%d')
timestr_filename = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

# Collect available tiffs, DEMS and points files
tiffs = [tiff for tiff in os.listdir(inbox) if tiff.endswith('.tif')]

for tiff in tiffs:
    tif_count +=1

    # check if corresponding DEM & points also exist, if so add to queue
    filename = os.path.splitext(tiff)

    dem_path = os.path.join(inbox, filename + '_DEM.tiff'))
    points_path = os.path.join(inbox, filename +  '.points'))

    if os.path.exists(dem_path) and os.path.exists(points_path):
        files.append(dict)




# define input, gcp, output files
input_file = r'D:\700 Georeferencing\Hendrik de Heer georeferencing/c07_hollandbean-Hendrik de Heer-20190503.tif'
gcp_file = r'D:\700 Georeferencing\Hendrik de Heer georeferencing/c07_hollandbean-Hendrik de Heer-20190503.points'
output_file = r'D:\700 Georeferencing\Hendrik de Heer georeferencing/c07_hollandbean-Hendrik de Heer-20190503_out.tif'

# Perform warp and translate ops
translate_and_warp_tiff(input_file, gcp_file, output_file)





## utilities and quick check ##

# https://gdal.org/drivers/raster/gtiff.html#overviews
# https://gdal.org/python/osgeo.gdal-module.html
# https://svn.osgeo.org/gdal/trunk/autotest/utilities/test_gdalwarp_lib.py

#translate_object.GetRasterBand(1).GetBlockSize()

#output_object.GetMetadata('IMAGE_STRUCTURE')
#if 'COMPRESSION' not in output_object or output_object['COMPRESSION'] != 'LZW':
#    gdaltest.post_reason('Did not get COMPRESSION')
