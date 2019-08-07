## Translate input geotiff + gcps -> output tiled geotiff in correct coordinate system
## optimised for multi-core, lots of RAM computer

# INPUT:
# (1) GeoTiff in SPS EPSG:4326
# (2) .points file generated with Georeferencer in QGIS, containing GCPs in format [mapX, mapY, destX, destY]  in EPSG:4326

# OUTPUT:
# (1) GeoTiff file georeferenced according to defined GCPs

import csv
from osgeo import gdal, osr
import time
import sys
import datetime
import os
import shutil

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

def translate_and_warp_tiff(input_file, gcp_file, output_file, filetype):

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

    
    # Set GDAL general and output config -- https://trac.osgeo.org/gdal/wiki/ConfigOptions
    # act as 'global' options, cleaned at  end of script
    
    gdaloptions = {'COMPRESS_OVERVIEW': 'JPEG', 
                   'PHOTOMETRIC_OVERVIEW': 'YCBR',
                   'INTERLEAVE_OVERVIEW': 'PIXEL',
                   'NUM_THREADS': 'ALL_CPUS'
                   }
    
    for key, val in gdaloptions.items():
        gdal.SetConfigOption(key,val)
    
    gdal.SetCacheMax = 3000

    # Set translation options for GDAL - hardcode reference system
    output_epsg=4326
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(output_epsg)
    
    # check if filetype is a DEM, use 32bit signed, otherwise 8-bit unsigned.
    if filetype == 'DEM':
        output_Type = gdal.GDT_Float32
    else:
        output_Type = gdal.GDT_Byte

    trnsopts = gdal.TranslateOptions(format='VRT',
                                     outputType=output_Type,
                                     outputSRS=dst_srs,
                                     GCPs=gcp_list
                                     )

    # Perform translate operation with GDAL -> output is VRT stored in system memory
    try:
        
        tic = time.time()
        translate_object = gdal.Translate('', input_object, options = trnsopts)
        toc = time.time()
        print('Translated orthomosaic in', (toc-tic), 'seconds')
    except:
        print('Failed to perform translate operation')
            
    # check if output is generated
    if translate_object is None:
        print('Translate object returned None, check input')

    # remote input file & object from working memory
    input_object = None

    # based on no of GCPs present, define transformation algorithm
    if len(gcp_list) < 10:
        tps_flag = False
    else:
        tps_flag = True

    # Set warping options for GDAL
    warpopts = gdal.WarpOptions(format='GTiff',
                                outputType=output_Type,
                                workingType=output_Type,
                                srcSRS=dst_srs,
                                dstSRS=dst_srs,
                                dstAlpha=True,
                                warpOptions=['NUM_THREADS=ALL_CPUS'],
                                warpMemoryLimit=3000,
                                creationOptions=['COMPRESS=LZW','TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS', 'JPEG_QUALITY=100', 'BIGTIFF=YES', 'ALPHA=YES'],
                                resampleAlg='cubicspline',
                                multithread=True,
                                tps=tps_flag,
                                transformerOptions=['NUM_THREADS=ALL_CPUS']
                                )

    # Perform actual warping operation -> output to specified path, filename
    try:
        tic = time.time()
        output_object = gdal.Warp(output_file, translate_object, options = warpopts)
        toc = time.time()
        print('Warped orthomosaic in', (toc-tic), 'seconds')
    except:
        print('Failed to perform warp operation')

    # check if output is generated
    if output_object is None:
        print('Warp object returned None, check input')
    
    # delete translate object, create some memory space
    translate_object = None
    
    # build internal overviews in compressed JPEG, only if ortho is processed
    try:
        tic = time.time()
        output_object.BuildOverviews("NEAREST", [8,16,32,64,128])
        toc = time.time()
        print('Added overviews to GeoTiff in', (toc-tic), 'seconds')
    except:
        print('Could not add internal overviews to output file')

    # clear remaining object(s) and gdal settings at end of script
    output_object = None
    
    for key, val in gdaloptions.items():
        gdal.SetConfigOption(key, None)
    

    return

## CONFIG SECTION ##
inbox = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\1_ready_to_rectify' #folder where all files are read from
path_ready_to_upload = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\2_ready_to_upload' #folder where all rectified files are stored
path_trashbin_originals = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\00_trashbin_originals' #folder where all rectified files are stored
path_rectified_DEMs_points = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\00_rectified_DEMs_points' #folder where all rectified files are stored


 #initialize
files = []
ortho_count = 0
duplicate = 0
timestr = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
datestr = datetime.datetime.now().strftime('%Y%m%d')
timestr_filename = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

# Collect available tiffs, DEMS and points files
tiffs = [tiff for tiff in os.listdir(inbox) if tiff.endswith('.tif')]
orthos = [s for s in tiffs if "_DEM.tif" not in s]

for ortho in orthos:
    ortho_count +=1

    # check if corresponding DEM & points also exist, if so add to queue
    filename = os.path.splitext(ortho)

    path_ortho = os.path.join(inbox, ortho)
    path_DEM = os.path.join(inbox, filename[0] + '_DEM.tif')
    path_points = os.path.join(inbox, filename[0] +  '.points')

    if os.path.exists(path_ortho) and os.path.exists(path_points):

        #name convention: customer-plot-date.tif
        this_customer_name,this_plot_name,this_datetime = ortho.split('-')
        this_datetime = this_datetime.split('.')[0]
        this_datetime = this_datetime.split('(')[0] #Splits of brackets for duplicates if present
        this_date = this_datetime[0:-4]
        this_time = this_datetime[-4::]

        path_ortho_out = os.path.join(path_ready_to_upload, filename[0] + '-GR.tif' )
        path_DEM_out = os.path.join(path_rectified_DEMs_points, filename[0] + '_DEM-GR.tif')

        queue = {"customer_name": this_customer_name, "plot_name":this_plot_name, "flight_date":this_date, "flight_time":this_time, "filename":filename[0],
                 "path_ortho": path_ortho, "path_DEM": path_DEM, "path_points": path_points,
                 "path_ortho_out": path_ortho_out, "path_DEM_out": path_DEM_out}
        ('    {}: {}\n'.format(str(ortho_count),filename))
        files.append(queue)

# queue of all flights for which an orthomosaic, DEM and .points file is available
process_queue = sorted(files, key=lambda k: k['flight_date'])

for file in process_queue:

    # georectify orthomosaic -> export to 'ready to upload folder'
    try:
        # translate_and_warp_tiff(input_file, gcp_file, output_file)
        translate_and_warp_tiff(file['path_ortho'], file['path_points'], file['path_ortho_out'], '')
        exported_tiff = True
    except:
        print('ortho rechtleggen werkt niet')
        
    # georectify DEM -> export to 'rectified DEMs folder'
    try:
        translate_and_warp_tiff(file['path_DEM'], file['path_points'], file['path_DEM_out'], 'DEM')
        exported_DEM = True
    except:
        print('dem rechtleggen werkt niet')

    # Move original (input) files to their respective folders
    if os.path.exists(path_ortho) and exported_tiff: # move original ortho
        shutil.move(file['path_ortho'], os.path.join(path_trashbin_originals,file['filename'] + '.tif'))

    if os.path.exists(path_DEM) and exported_DEM: # move original DEM
        shutil.move(file['path_DEM'], os.path.join(path_trashbin_originals,file['filename'] + '_DEM.tif'))

    if os.path.exists(path_points): # move used points file
        if exported_DEM or exported_tiff:
            shutil.move(file['path_points'], os.path.join(path_rectified_DEMs_points,file['filename'] + '.points'))



## utilities and quick check ##

#translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-of Gtiff -co COMPRESS=LZW"))
#gdal.Translate(gdaloutput, gdalinput, options=translateoptions)

# https://gdal.org/drivers/raster/gtiff.html#overviews
# https://gdal.org/python/osgeo.gdal-module.html
# https://svn.osgeo.org/gdal/trunk/autotest/utilities/test_gdalwarp_lib.py

#translate_object.GetRasterBand(1).GetBlockSize()

#output_object.GetMetadata('IMAGE_STRUCTURE')
#if 'COMPRESSION' not in output_object or output_object['COMPRESSION'] != 'LZW':
#    gdaltest.post_reason('Did not get COMPRESSION')
