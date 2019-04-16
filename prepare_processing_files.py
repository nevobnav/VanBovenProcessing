# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:21:17 2019

@author: VanBoven (Eric)
"""

#check for new uploads every x minutes and put in line for processing
#resulting output are text files with image path and names per plot or text files with images that are not linked to a flight/plot and should not be processed
"""
1. get a list of all the folders with new uploaded images (every 15 min for example while not processing)
2. check if all folders are finished with uploading
3. Sort the list based on time in .exit file
4. Split the groups of images per parcel using geometry of parcel
5. Start processing in agisoft per parcel
6. Export orthomosaic in folder of parcel
7. Copy/cut images and move to archive
"""

import os, re
import glob
import datetime
import time
import pandas as pd
import numpy as np
import PIL.ExifTags
import PIL.Image
import psycopg2
import json
import logging

from shapely.geometry import mapping
from shapely.geometry import Point
from shapely.wkb import loads, dumps
import geopandas as gpd

from sqlalchemy import create_engine, MetaData, Integer, ForeignKey, DateTime, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape

from functools import partial
import pyproj
from shapely.ops import transform

#initiate log file
timestr = time.strftime("%Y%m%d-%H%M%S")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = r"E:\VanBovenDrive\VanBoven MT\Processing\Log_files/" + str(timestr) + "_processing_files_log_file" +  ".log",level=logging.DEBUG)


# variables
root_path = r'E:\VanBovenDrive\VanBoven MT\Opnames'
#steps_to_uploads is the number of folders starting from the root drive untill the uploads folder
#for example: in the folder "E:\VanBovenDrive\VanBoven MT\Opnames\c04_verdegaal\20190304" the steps_to_uploads = 6
steps_to_uploads = 6
#the last number of days to process, standard only the last week is considered for new uploads
nr_of_days_to_process = 7
#the maximum time in seconds allowed between images to be considered from the same flight
max_time_diff = 1800
#minimum nr of images per ha needed to process a flight
min_nr_of_images_per_ha = 30
#db connection info
config_file_path = r'C:\Users\VanBoven\MijnVanBoven\config.json'
port = 5432

#steps_to_uploads is the number of folders starting from the root drive untill the uploads folder
#for example: in the folder "E:\VanBovenDrive\VanBoven MT\Opnames\c04_verdegaal\20190304" the steps_to_uploads = 6

def getListOfFolders(root_path, steps_to_uploads, nr_of_days_to_process):
    #get a list of all folders
    folderList = pd.DataFrame([x[0] for x in os.walk(root_path)], columns = ['Path'])
    #select all folders with new uploaded images
    nr_of_subdirs = folderList.Path.str.split("\\")
    nr_of_subdirs = nr_of_subdirs.apply(lambda x:len(x))
    folderList['Nr_of_subdirs'] = nr_of_subdirs
    uploads = folderList[folderList.Nr_of_subdirs == steps_to_uploads]
    uploads['Day'] = uploads['Path'].apply(lambda x:os.path.basename(x))
    #get day of upload as datetime object
    uploads['Date'] = uploads['Day'].apply(lambda x:pd.to_datetime(x, format = '%Y%m%d'))
    #get the last 7 days of uploading
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=nr_of_days_to_process)
    new_uploads = uploads[uploads['Date'] > week_ago]
    #continue only when there are new uploads
    if len(new_uploads) > 0:
        #check for each folder if uploading is finished and if the files have not been processed
        new_uploads['Finished'] = new_uploads.Path.apply(lambda x:bool([True for y in os.listdir(x) if y.endswith('exit.txt')]))
        new_uploads['Processed'] = new_uploads.Path.apply(lambda x:bool([True for y in os.listdir(x)if y.endswith('processed.txt')]))
        #return only folders with finished uploads
        new_finished_uploads = new_uploads[(new_uploads['Finished'] == True) & (new_uploads['Processed'] == False)]
        #continue only when new uploads have finished uploading
        if len(new_finished_uploads) > 0:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            logging.info(str(len(new_finished_uploads)) + " new uploads at " + str(timestr))
            logging.info("\n")
            return new_finished_uploads
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            logging.info("No new finished uploads at " + str(timestr))
            logging.info("\n")
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        logging.info("No new uploads at " + str(timestr))
        logging.info("\n")

def CreateProcessingOrderUploads(new_finished_uploads):
    #initiate lists to store relevant values per upload event, metashape needs a list of image paths as input.
    #the time finished uploading is used to determine the processing order, and the number of images to determine if the number of images reflects a complete plot
    time_finished = []
    image_count = []
    folderList = []
    image_names = []
    #loop through folders with finished uploads and open init and exit metadata files
    for folder in new_finished_uploads.Path:
        for file in os.listdir(folder):
        #extract the number of uploaded images and the time when uploading was finished from exit metadata file
            if file.endswith('exit.txt'):
                link = os.path.join(folder, file)
                with open(link) as fp:
                    for i, line in enumerate(fp):
                        if (line.startswith('Time')) and (i < 3):
                            date_time = line.replace('Time: ', '').rstrip()
                            date_time_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
                            time_finished.append(date_time_obj)

                        elif (line.startswith('Successful uploads')) or (line.startswith('Succesfull uploads')):
                            upload_info = line[20:-3]
                            upload_numbers = [int(s) for s in upload_info.split() if s.isdigit()]
                            try:
                                successful_uploads, all_images = upload_numbers
                                #check if upload was successful and log info otherwise
                                if all_images - successful_uploads > 1:
                                    missing_uploads = all_images - successful_uploads
                                    with open(r'C:\Users\VanBoven\Documents\Temp_processing_files\Error_logs\upload_error_in_' +str(link[58:-9])+".txt", "w") as text_file:
                                        text_file.write("There are " + str(missing_uploads) + " unsuccessful uploads in " + folder)
                            except:
                                logging.exception(".exit is not compatible with current format")

                            #currently images are still processed, despite missing uploads
                        elif (i > 4):
                            break
        #beun oplossing, hier moet alleen jpg geteld worden
        image_count.append(len(os.listdir(folder))-2)
        list_of_images = ([x for x in os.listdir(folder) if x.endswith('.JPG')])  #glob.glob(folder + '/*.JPG')
        folderList.append(folder)
        image_names.append(list_of_images)
        if len(time_finished) < len(folderList):
            time_finished.append(datetime.date.today() + datetime.timedelta(days=1))
        #if len(time_finished) < 1:
            #time_finished.append(datetime.date.today() + datetime.timedelta(days=1))
    #Create dataframe with all information for processing and sort based on datetime
    files_to_process = pd.DataFrame({'Path': folderList, 'Image_names': image_names, 'Time_finished_uploading': time_finished, 'Image_count': image_count})
    files_to_process['Processing_order'] = files_to_process['Time_finished_uploading'].rank(ascending=1)
    files_to_process = files_to_process.set_index(files_to_process['Time_finished_uploading'])
    files_to_process = files_to_process.sort_index(ascending=1)
    return files_to_process

#function to access image metadata
def getExif(img):
    exif = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img._getexif().items()
    if k in PIL.ExifTags.TAGS
    }
    return exif

def get_image_coords(folder, img):
    gps_info = getExif(PIL.Image.open(os.path.join(folder, img))).get('GPSInfo')
    d, m, s = gps_info[2]
    #get degrees, minutes, seconds for lat
    d_lat = d[0]/d[1]
    m_lat = m[0]/m[1]
    s_lat = s[0]/s[1]
    #get degrees, minutes secondes for lon
    d, m, s = gps_info[4]
    d_lon = d[0]/d[1]
    m_lon = m[0]/m[1]
    s_lon = s[0]/s[1]
    #convert to decimal degrees
    dd_lat = d_lat + float(m_lat)/60 + float(s_lat)/3600
    dd_lon = d_lon + float(m_lon)/60 + float(s_lon)/3600
    #create point geometry
    coord = (dd_lon, dd_lat)
    return coord

def connect(config_file_path, port):
    '''Returns a connection and a metadata object'''

    with open(config_file_path) as config_file:
        config = json.load(config_file)

    host=config.get("DB_IP")
    db=config.get("DB_NAME")
    user=config.get("DB_USER")
    password=config.get("DB_PASSWORD")

    # We connect with the help of the PostgreSQL URL
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    con = create_engine(url, client_encoding='utf8')

    # We then bind the connection to MetaData()
    meta = MetaData(bind=con, reflect=True)

    return con, meta

def get_customer_pk(customer_id,meta,con):
    customers = meta.tables['portal_customer']
    query = select([customers.c.id])
    query = query.where(customers.c.customer_name == customer_id)
    res = con.execute(query)
    for result in res:
        customer_pk = result[0]
    return customer_pk

def get_customer_plots(customer_id, meta, con):
    customer_pk = get_customer_pk(customer_id,meta,con)
    plots = meta.tables['portal_plot']
    query_name = select([plots.c.name])
    query_name = query_name.where(plots.c.customer_id == customer_pk)
    query_id = select([plots.c.id])
    query_id = query_id.where(plots.c.customer_id == customer_pk)
    res_name = con.execute(query_name)
    res_id = con.execute(query_id)
    plot_names = []
    plot_ids = []
    for result in res_name:
        new_val = result[0]
        plot_names.append(new_val)
    for result in res_id:
        new_val = result[0]
        plot_ids.append(new_val)
    return plot_names, plot_ids

def get_plot_shape(plot_id, meta,con):
    plots = meta.tables['portal_plot']
    query= select([plots.c.shape])
    query = query.where(plots.c.id == plot_id)
    res = con.execute(query)
    for result in res:
        output = result[0]
    return output

def transform_geometry(geometry):
    project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:4326'), # source coordinate system
    pyproj.Proj(init='epsg:28992')) # destination coordinate system
    geometry = transform(project, geometry)  # apply projection
    return geometry

def GroupImagesPerPlot(files_to_process, max_time_diff, min_nr_of_images_per_ha, con, meta):
    #loop through folders to process
    for z in range(len(files_to_process)):
        folder = files_to_process['Path'].iloc[z]
        dirs = folder.split("\\")
        #Get the customer_id
        get_customer_name = [re.search('c' + '\d+' + '_' + '.*', subdir) for subdir in dirs]
        customer = [customer.group(0) for customer in get_customer_name if customer]
        customer_id = customer[0]
        #get recoring date
        date_of_recording = os.path.basename(folder)
        #get the image names and path
        images = pd.DataFrame({'Image_names':files_to_process['Image_names'].iloc[z]})
        #read metadata
        metadata = pd.DataFrame({'Metadata':images['Image_names'].apply(lambda x:getExif(PIL.Image.open(os.path.join(folder, x))))})
        #get altitude from Exif data
        #images['Altitude'] = images['Image_names'].apply(lambda x:pd.Series({'Alt':(getExif(PIL.Image.open(os.path.join(folder, x))).get('GPSInfo')[6][0])/(getExif(PIL.Image.open(os.path.join(folder, x))).get('GPSInfo')[6][1])}))
        images['Altitude'] = metadata['Metadata'].apply(lambda x:pd.Series({'Alt':(x.get('GPSInfo')[6][0])/(x.get('GPSInfo')[6][1])}))
        #get the coordinates of the images from the metadata this function has to be improved to used metadata dataframe instead of image metadata with PIL
        images['Coords'] = images['Image_names'].apply(lambda x:pd.Series({'Coords':Point(get_image_coords(folder, x))}))
        #get the date and time of the images from the metadata
        images['DateTime'] = metadata['Metadata'].apply(lambda x:pd.to_datetime(x.get('DateTime'), format = '%Y:%m:%d %H:%M:%S'))
        images['ISO'] = metadata['Metadata'].apply(lambda x:pd.Series({'ISO':(x.get('ISOSpeedRatings'))}))
        images['Exposure_time'] = metadata['Metadata'].apply(lambda x: pd.Series({'Exposure':int(x.get('ExposureTime')[1]/x.get('ExposureTime')[0])}))
        images = images.sort_values(by='DateTime', ascending=True)
        images = images.reset_index(drop=True)
        #Group images from the same flights
        #images['Groupby_nr'] = np.where(((images['Time_after_previous'] > max_time_diff ) & (images['Time_before_next'] < max_time_diff)) | ((images['Time_after_previous'] > max_time_diff ) & (images['Time_before_next'] > max_time_diff)),1,0).cumsum()
        images['Input_folder'] = images['Image_names'].apply(lambda x:os.path.join(folder, x))
        #log the number of image groups for possible debugging
        timestr = time.strftime("%Y%m%d-%H%M%S")
        #logging.info(str(images['Groupby_nr'].max()) + " image groups at " + str(timestr))
        total_upload = images.copy()
        exp_test = pd.DataFrame({'exp_test':images['Exposure_time'] == 320})
        iso_min = images['ISO'].min()
        iso_max = images['ISO'].max()
        if not exp_test['exp_test'].all():
            with open(r"C:\Users\VanBoven\Documents\100 Ortho Inbox/" + str(customer_id) + '_' + str(date_of_recording)+".txt", 'a') as f:
                f.write(str('Not all images have exposure of 1/320\n'))
                f.write('Range of iso values = '+str(iso_min)+'-'+str(iso_max)+'\n')
        if iso_max > 800:
            with open(r"C:\Users\VanBoven\Documents\100 Ortho Inbox/" + str(customer_id) + '_' + str(date_of_recording)+".txt", 'a') as f:
                f.write(str('Iso value was > 800, check quality and reason.\n'))
                f.write('Range of iso values = '+str(iso_min)+'-'+str(iso_max)+'\n')

        #get plots of customer from DB
        pk = get_customer_pk(customer_id,meta,con)
        plot_names, plot_ids = get_customer_plots(customer_id, meta, con)

        #check per plot for intersecting images and create a file for processing
        for i, plot_id in enumerate(plot_ids):
            #create copy of images for because the merge step ads column to the file
            temp = images.copy()
            #convert wkb element to shapely geometry and create a buffer around the shape of approx. 10 meters (unit is decimal degrees)
            geometry = to_shape(get_plot_shape(plot_id, meta, con))
            geometry_transformed = transform_geometry(geometry)
            geometry_buffered = geometry.buffer(0.0001)
            #select intersecting images
            intersect = pd.DataFrame({str(plot_names[i]):total_upload['Coords'].apply(lambda x:geometry_buffered.contains(x))})
            plot_name = plot_names[i]
            total_upload[str(plot_names[i])] = total_upload['Coords'].apply(lambda x:geometry_buffered.contains(x))
            output = pd.DataFrame(pd.merge(temp, intersect, left_index = True, right_index = True))
            output = output[output[str(plot_name)] == True]
            if len(output)>0:
                output = pd.DataFrame(output[output[str(plot_names[i])] == True])
                output['Altitude_difference'] = output['Altitude'].diff()
                output['Time_after_previous'] = output['DateTime'].diff().astype('timedelta64[s]')
                output['Time_before_next'] = output['DateTime'].shift(-1).diff().astype('timedelta64[s]')
                output['Groupby_nr'] = np.where((abs(output['Altitude_difference']) > 18),1,0).cumsum()
                #Loop through clustered_images per plot
                multiple_outputs = []
                for j in range(output['Groupby_nr'].max()+1):
                    subset = pd.DataFrame(output[output['Groupby_nr'] == j])
                    if len(subset) < 10:
                        output.drop(output[output['Groupby_nr'] == j].index, inplace = True)
                    if (len(subset) > geometry_transformed.area * 0.0001 * min_nr_of_images_per_ha) and (subset['Time_before_next'].max() > max_time_diff):
                        multiple_outputs.append(output[output['Groupby_nr'] <= j])
                        output = output[output['Groupby_nr'] > j]
                if len(output) > geometry_transformed.area * 0.0001 * min_nr_of_images_per_ha:
                    multiple_outputs.append(output)
                    #rep = {"Opnames": "Archive", str(customer_id): str(customer_id+"\\"+str(plot_name))}
                    # use these three lines to do the replacement
                    #rep = dict((re.escape(k), v) for k, v in rep.items())
                    #pattern = re.compile("|".join(rep.keys()))
                    #output['Output_folder'] = output['Input_folder'].apply(lambda x:os.path.join(pattern.sub(lambda m: rep[re.escape(m.group(0))], os.path.dirname(x))))#, os.path.basename(x)))
                    #create txt file for processing
                if len(multiple_outputs) > 0:
                    for output in multiple_outputs:
                        time.sleep(2)
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        #output['Output_folder'] = output['Input_folder'].apply(lambda x:pattern.sub(lambda m: rep[re.escape(m.group(0))], x))
                        output[['Input_folder']].to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_' + str(date_of_recording) + '_' + str(customer_id) + '_' + str(plot_name)+".txt", sep = ',', header = False, index = False)
                        with open(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_' + str(date_of_recording) + '_' + str(customer_id) + '_' + str(plot_name)+".txt", 'a') as f:
                            f.write(str(plot_name))
                    #output[['Input_folder', 'Output_folder']].to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_' + str(customer_id) + '_' + str(plot_name)+".txt", sep = ',', header = False, index = False)
        image_matches = total_upload.select_dtypes(np.bool)
        unknown_plot = pd.DataFrame({'Plot_known' : image_matches.any(axis=1, bool_only = True)})
        unknown_plot = unknown_plot[unknown_plot['Plot_known'] == False]
        unknown_plot = pd.merge(total_upload, unknown_plot, left_index = True, right_index = True)
        if len(unknown_plot) > 50:
            #if true, create a processing file with unknown plot and correct output folder
            #rep = {"Opnames": "Archive", str(customer_id): str(customer_id+"\\unknown_plot_id")}
            # modify input folder to output folder
            #rep = dict((re.escape(k), v) for k, v in rep.items())
            #pattern = re.compile("|".join(rep.keys()))
            #unknown_plot['Output_folder'] = unknown_plot['Input_folder'].apply(lambda x:pattern.sub(lambda m: rep[re.escape(m.group(0))], x))
            unknown_plot[['Input_folder']].to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_'+ str(customer_id) +'_unknown_plot.txt', sep = ',', header = False, index = False)
            with open(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_' + str(customer_id) + '_unknown_plot.txt', 'a') as f:
                f.write('unknown_plot')
        for plot_name in plot_names:
            total_upload.loc[total_upload[str(plot_name)] == True, 'Plot'] = str(plot_name)
            total_upload = total_upload.drop(columns=[str(plot_name)])
        total_upload['Altitude_difference'] = total_upload['Altitude'].diff()
        total_upload['Time_after_previous'] = total_upload['DateTime'].diff().astype('timedelta64[s]')
        total_upload['Time_before_next'] = total_upload['DateTime'].shift(-1).diff().astype('timedelta64[s]')
        total_upload['Groupby_nr'] = np.where((abs(total_upload['Altitude_difference']) > 18),1,0).cumsum()
        total_upload.to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\Log_files/" + timestr + '_' + str(customer_id) +"_image_groups.csv")

    """
        #check for images
        #Loop through images per flight
        for j in range(images['Groupby_nr'].max()):
            flight = pd.DataFrame(images[images['Groupby_nr'] == j])
            #check if group of images is significant nr
            if len(flight) > min_nr_of_images:
                #get plots of customer from DB
                pk = get_customer_pk(customer_id,meta,con)
                plots = get_customer_plots(customer_id, meta, con)
                #check per plot for intersecting images and create a file for processing
                for plot_id in plots:
                    #convert wkb element to shapely geometry and create a buffer around the shape of approx. 10 meters (unit is decimal degrees)
                    geometry = to_shape(get_plot_shape(plot_id, meta, con)).buffer(0.0001)
                    #select intersecting images
                    flight[str(plot_id)] = flight['Coords'].apply(lambda x:geometry.contains(x))
                    output = pd.DataFrame(flight['Input_folder'].loc[flight[str(plot_id)] == True])
                    #check again if the intersecting nr of images is enough for processing
                    if len(output) > min_nr_of_images:
                        #set output folder based on input folder, customer id and plot id and folder structure on drive
                        rep = {"Opnames": "Archive", str(customer_id): str(customer_id+"\\"+str(plot_id))}
                        # use these three lines to do the replacement
                        rep = dict((re.escape(k), v) for k, v in rep.items())
                        pattern = re.compile("|".join(rep.keys()))
                        output['Output_folder'] = output['Input_folder'].apply(lambda x:pattern.sub(lambda m: rep[re.escape(m.group(0))], x))
                        #create txt file for processing
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        output.to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_' + str(customer_id) + '_' + str(plot_id)+ "_group"+str(j)+".txt", sep = ',', header = False, index = False)
                    else:
                        #if nr of images within plot is not enough for processing, the output column value is put back to False, as if the images do not intersect a plot
                        flight[str(plot_id)].loc[flight[str(plot_id)] == True] = False
                #Get list of images that fall not within a single plot of the customer
                img_check = flight.drop(['Image_names', 'Altitude', 'Coords', 'DateTime', 'Altitude_difference',
                                         'Time_after_previous', 'Time_before_next', 'Groupby_nr', 'Input_folder'],axis = 1)
                #check for images that did not intersect with any plot
                z = img_check.any(axis = 'columns')
                unknown_plot = pd.merge(pd.DataFrame(z.loc[z == False]), flight, how = 'left', left_index = True, right_index = True)
                #check if the number of images is enough to process, assumption here is that images within a flight are related
                if len(unknown_plot) > min_nr_of_images:
                    #if true, create a processing file with unknown plot and correct output folder
                    rep = {"Opnames": "Archive", str(customer_id): str(customer_id+"\\unknown_plot_id")}
                    # modify input folder to output folder
                    rep = dict((re.escape(k), v) for k, v in rep.items())
                    pattern = re.compile("|".join(rep.keys()))
                    unknown_plot['Output_folder'] = unknown_plot['Input_folder'].apply(lambda x:pattern.sub(lambda m: rep[re.escape(m.group(0))], x))
                    unknown_plot[['Input_folder','Output_folder']].to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_process/" + timestr + '_'+ str(customer_id) + "_group"+str(j)+'_unknown_plot.txt', sep = ',', header = False, index = False)
                else:
                    #if not, create a file to move/remove images from recording folder on drive
                    rep = {"Opnames": "Archive", str(customer_id): str(customer_id+"\\random_images")}
                    # modify input folder to output folder
                    rep = dict((re.escape(k), v) for k, v in rep.items())
                    pattern = re.compile("|".join(rep.keys()))
                    unknown_plot['Output_folder'] = unknown_plot['Input_folder'].apply(lambda x:pattern.sub(lambda m: rep[re.escape(m.group(0))], x))
                    unknown_plot[['Input_folder', 'Output_folder']].to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_move/" + timestr + '_'+ str(customer_id) + "_group"+str(j)+'_not_enough_images.txt', sep = ',', header = False, index = False)
            else:
                #create a file to move images from recordings folder on drive
                rep = {"Opnames": "Archive", str(customer_id): str(customer_id+"\\random_images")}
                # modify input folder to output folder
                rep = dict((re.escape(k), v) for k, v in rep.items())
                pattern = re.compile("|".join(rep.keys()))
                flight['Output_folder'] = flight['Input_folder'].apply(lambda x:pattern.sub(lambda m: rep[re.escape(m.group(0))], x))
                flight[['Input_folder', 'Output_folder']].to_csv(r"E:\VanBovenDrive\VanBoven MT\Processing\To_move/" + timestr + '_'+ str(customer_id) + "_group"+str(j)+'_random_images.txt', sep = ',', header = False, index = False)
"""

def processing(root_path, steps_to_uploads, max_time_diff, min_nr_of_images_per_ha, config_file_path, port, nr_of_days_to_process):
    new_finished_uploads = getListOfFolders(root_path, steps_to_uploads, nr_of_days_to_process)
    if new_finished_uploads is not None:
        files_to_process = CreateProcessingOrderUploads(new_finished_uploads)
        con, meta = connect(config_file_path, port)
        GroupImagesPerPlot(files_to_process, max_time_diff, min_nr_of_images_per_ha, con, meta)

try:
    processing(root_path, steps_to_uploads, max_time_diff, min_nr_of_images_per_ha, config_file_path, port, nr_of_days_to_process)
except Exception:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.info("Error encountered at the following time: " + str(timestr))
    logging.exception("No processing due to the following error:")
    logging.info("\n")
