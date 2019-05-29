#Batch processing script version 1.0

"""
1. Loading images
    - All images in a folder have to be processed in one go, multiple folders have to be processed subsequently
    - nice to have: check image quality and remove bad images based on agisoft metric
2. Update altitude metadata
3. Align cameras
    - nice to have: Optimize camera iteratively based on reprojection error and gradual selection
    - export tie points for recycling in HQ run.
4. Dense cloud calculation
5. Create dem
6. Create orthomosaic
7. export orthomosaic
"""

"""
updates voor v1.1:
    - Images will be archived as upload event.They will be left in the recordings folder.
    - After processing a file is written to the folder to indicate that the files have been processed
    - Benefit is that the folder with images as a basis will always exist
    - when processing, filter uploads to contain only uploads of for example last week. (to reduce time of iterating trough folders)

"""


import Metashape
import os,re,sys
import time
import logging
import shutil
import pandas as pd
from append_df_to_excel_file import *

def get_scan_information(chunk):
    #sensor type (str)
    sensor = chunk.sensors[0]
    camera = chunk.cameras[1]
    #flying altitude (float)
    height_of_flight = camera.reference.location.z
    ortho = chunk.orthomosaic
    #gsd in meters (float)
    gsd = ortho.resolution
    #flight_datetime format: '2019:04:30 19:59:56'
    flight_datetime = camera.photo.meta['Exif/DateTime']
    #zoomlevel op basis van gsd, de grenzen zijn nu gekozen zodat 30/35 meter nog op zoomniveau 23 wordt getiled en eventuele lagere vluchten op 24/25
    zoomlevel = 23
    if gsd < 0.007 and gsd > 0.005:
         zoomlevel = 24
    if gsd < 0.005:
        zoomlevel = 25
    return (height_of_flight, gsd, zoomlevel, flight_datetime, sensor)

def move_files_after_processing(photoList, output_folder):
    for photo in photoList:
        shutil.copy(photo, output_folder)
        #shutil.move(photo, output_folder)
    fileList = os.listdir(os.path.dirname(photoList[0]))
    if len(fileList) > 0:
        for file in fileList:
            shutil.move(os.path.join(os.path.dirname(photoList[0]), file), output_folder)
    #can give trouble with permission in windows
    #os.removedirs(os.path.dirname(photoList[0]))

def getAltitude(chunk):
    for camera in chunk.cameras:
    	if "DJI/RelativeAltitude" in camera.photo.meta.keys() and camera.reference.location:
    		z = float(camera.photo.meta["DJI/RelativeAltitude"])
    		camera.reference.location = (camera.reference.location.x, camera.reference.location.y, z)
    chunk.updateTransform()
    Metashape.app.update()
    print("Script finished")

def get_first_img_time(chunk):
    camera = chunk.cameras[0]
    cam_datetime = camera.photo.meta['Exif/DateTime']
    img_time = cam_datetime[11:16].replace(':','')
    return img_time

def MetashapeProcess(photoList, day_of_recording, metashape_processing_folder, ortho_out, quality):
    #if folder.endswith('Perceel1'): ook een optie afhankelijk van naamgeving mappen
    #path = folder
    if not os.path.exists(metashape_processing_folder):
        os.makedirs(metashape_processing_folder)
    #start with cleared console
    Metashape.app.console.clear()

    ## construct the document class
    doc = Metashape.app.document

    ## save project
    #doc.open("M:/Metashape/practise.psx")
    timestr_save = time.strftime("%Y%m%d-%H%M%S")
    psxfile = metashape_processing_folder + '\\' + str(day_of_recording) + "_" + str(timestr_save)+'.psx'
    doc.save( psxfile )
    print ('&amp;gt;&amp;gt; Saved to: ' + psxfile)

    ## point to current chunk
    #chunk = doc.chunk

    ## add a new chunk
    chunk = doc.addChunk()

    ## set coordinate system
    # - Metashape.CoordinateSystem("EPSG::4612") --&amp;gt; JGD2000
    chunk.crs = Metashape.CoordinateSystem("EPSG::4326")

    ################################################################################################
    ### get photo list ###
    #photoList = []
    #getPhotoList(path, photoList)
    #print (photoList)
    #photoList = getPhotoList(path)
    ################################################################################################
    ### add photos ###
    # addPhotos(filenames[, progress])
    # - filenames(list of string) – A list of file paths.
    chunk.addPhotos(photoList)
    getAltitude(chunk)
    #set ground altitude at 0
    chunk.meta["ground_altitude"] = "0"
    chunk.camera_rotation_accuracy = [10.0,5.0,5.0]
    ################################################################################################
    ### align photos ###
    ## Perform image matching for the chunk frame.
    # - Alignment accuracy in [HighestAccuracy, HighAccuracy, MediumAccuracy, LowAccuracy, LowestAccuracy]
    # - Image pair preselection in [ReferencePreselection, GenericPreselection, NoPreselection]
    tic = time.clock()
    chunk.matchPhotos(accuracy=Metashape.HighestAccuracy, preselection=Metashape.ReferencePreselection, filter_mask=False, keypoint_limit=40000, tiepoint_limit=4000)
    chunk.alignCameras(adaptive_fitting=True)

    #iteratively align images until at least 97% is aligned
    alignment_check = 1
    max_iter = 10
    iter = 0
    while alignment_check > 0.02:
        realign_list = list()
        for camera in chunk.cameras:
            if not camera.transform:
                realign_list.append(camera)
        if (len(realign_list)/len(chunk.cameras) > 0):
            chunk.alignCameras(cameras = realign_list)
        alignment_check = len(realign_list)/len(chunk.cameras)
        iter += 1
        if iter == max_iter:
            break
    toc = time.clock()
    processing_time = toc - tic
    logging.info("Image alignment took " + str(int(processing_time)) + " seconds" )
    logging.info("alignment took "+str(iter) + " iterations")
    logging.info("A total of " + str(len(chunk.cameras) - len(realign_list)) + " out of " + str(len(chunk.cameras)) + " images has been aligned")

    tic = time.clock()
    #optional incteratively increase camera accuracy
    threshold = 0.5
    f = Metashape.PointCloud.Filter()
    f.init(chunk, criterion = Metashape.PointCloud.Filter.ReprojectionError)
    f.removePoints(threshold)
    #optimize cameras based on accurate points
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True, fit_k1=True,
    fit_k2=True, fit_k3=True, fit_k4=False, fit_p1=True, fit_p2=True, fit_p3=False,
    fit_p4=False, adaptive_fitting=False, tiepoint_covariance=False)
    toc = time.clock()
    processing_time = toc - tic
    logging.info("Camera optimization took "+str(int(processing_time)) + " seconds")

    ################################################################################################
    ### build dense cloud ###
    ## Generate depth maps for the chunk.
    # buildDenseCloud(quality=MediumQuality, filter=AggressiveFiltering[, cameras], keep_depth=False, reuse_depth=False[, progress])
    # - Dense point cloud quality in [UltraQuality, HighQuality, MediumQuality, LowQuality, LowestQuality]
    #every step lower dan UltraQuality downscales the images by a factor 4 (2x per side)
    # - Depth filtering mode in [AggressiveFiltering, ModerateFiltering, MildFiltering, NoFiltering]
    tic = time.clock()

    if quality == "Medium":
        chunk.buildDepthMaps(quality=Metashape.MediumQuality, filter=Metashape.MildFiltering)
    if quality == "Low":
        chunk.buildDepthMaps(quality=Metashape.LowQuality, filter=Metashape.MildFiltering)
    chunk.buildDenseCloud(max_neighbors = 100, point_colors = False)
    toc = time.clock()
    processing_time = toc - tic
    logging.info("Dense cloud generation took " + str(int(processing_time)) + " seconds")

    ################################################################################################
    ### build mesh ###
    ## Generate model for the chunk frame.
    # buildModel(surface=Arbitrary, interpolation=EnabledInterpolation, face_count=MediumFaceCount[, source ][, classes][, progress])
    # - Surface type in [Arbitrary, HeightField]
    # - Interpolation mode in [EnabledInterpolation, DisabledInterpolation, Extrapolated]
    # - Face count in [HighFaceCount, MediumFaceCount, LowFaceCount]
    # - Data source in [PointCloudData, DenseCloudData, ModelData, ElevationData]
    #chunk.buildModel(surface=Metashape.HeightField, interpolation=Metashape.EnabledInterpolation, face_count=Metashape.HighFaceCount)

    ################################################################################################
    ### build texture (optional) ###
    ## Generate uv mapping for the model.
    # buildUV(mapping=GenericMapping, count=1[, camera ][, progress])
    # - UV mapping mode in [GenericMapping, OrthophotoMapping, AdaptiveOrthophotoMapping, SphericalMapping, CameraMapping]
    #chunk.buildUV(mapping=Metashape.AdaptiveOrthophotoMapping)
    ## Generate texture for the chunk.
    # buildTexture(blending=MosaicBlending, color_correction=False, size=2048[, cameras][, progress])
    # - Blending mode in [AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending]
    #chunk.buildTexture(blending=Metashape.MosaicBlending, color_correction=True, size=30000)

    ################################################################################################
    ## save the project before build the DEM and Ortho images
    doc.save(psxfile)

    ################################################################################################
    ### build DEM (before build dem, you need to save the project into psx) ###
    ## Build elevation model for the chunk.
    # buildDem(source=DenseCloudData, interpolation=EnabledInterpolation[, projection ][, region ][, classes][, progress])
    # - Data source in [PointCloudData, DenseCloudData, ModelData, ElevationData]
    tic = time.clock()
    chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation, projection=chunk.crs)
    toc = time.clock()
    processing_time = toc - tic
    logging.info("DEM generation took "+str(int(processing_time))+" seconds")
    ################################################################################################
    ## Build orthomosaic for the chunk.
    # buildOrthomosaic(surface=ElevationData, blending=MosaicBlending, color_correction=False[, projection ][, region ][, dx ][, dy ][, progress])
    # - Data source in [PointCloudData, DenseCloudData, ModelData, ElevationData]
    # - Blending mode in [AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending]
    tic = time.clock()
    chunk.buildOrthomosaic(surface=Metashape.ElevationData, blending=Metashape.MosaicBlending, projection=chunk.crs)
    toc = time.clock()
    processing_time = toc - tic
    logging.info("Orthomosaic generation took "+str(int(processing_time))+" seconds")
    ################################################################################################
    ## auto classify ground points (optional)
    #chunk.dense_cloud.classifyGroundPoints()
    #chunk.buildDem(source=Metashape.DenseCloudData, classes=[2])

    ################################################################################################
    doc.save(psxfile)

    #if not os.path.exists(temp_processing_folder+"\\Orthomosaic\\"):
        #os.makedirs(temp_processing_folder + "\\Orthomosaic\\")

    #get time of first image
    img_time = get_first_img_time(chunk)

    timestr = time.strftime("%H%M%S")
    #zorg voor mooie naamgeving + output
    tic = time.clock()
    #check if output allready exists and rename if so
    ortho_out = ortho_out[:-4]+str(img_time)+'.tif'
    name_it = 1
    while os.path.isfile(str(ortho_out)) == True:
        ortho_out = ortho_out[:-4] + '('+str(name_it)+').tif'
        name_it += 1
    chunk.exportOrthomosaic(path = ortho_out, tiff_big = True) #, jpeg_quality(75)
    #finish and write to logfile
    toc = time.clock()
    processing_time = toc-tic
    logging.info("Ortho export took "+str(int(processing_time))+" seconds")
    doc.clear()
    
    scan_information = get_scan_information(chunk)
    return timestr_save, scan_information

#Start of execution
#get quality parameter from bat script
quality = sys.argv[1]

#initiate log file
timestr = time.strftime("%Y%m%d-%H%M%S")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = r"E:\VanBovenDrive\VanBoven MT\Processing\Log_files/" + str(timestr) + "_Metashape_log_file.log",level=logging.DEBUG)

#root_path = r'E:\100 Testing\190304 batch_script_test'
process_path = r'E:\VanBovenDrive\VanBoven MT\Processing\To_process'
move_path = r'E:\VanBovenDrive\VanBoven MT\Processing\To_move'
processing_archive_path = r'E:\VanBovenDrive\VanBoven MT\Processing\Archive'
processing_folder = r'C:\Users\VanBoven\Documents\100 Ortho Inbox'
temp_processing_folder = r'E:\Metashape'
#execute:

#keep track of processing
nr_of_plots = 0
nr_of_images = 0
#iterate through the folder with processing txt files
for proces_file in os.listdir(process_path):
    if proces_file.endswith('.txt'):
        try:
            input_file = os.path.join(process_path, proces_file)
            with open(input_file) as image_file:
                temp = image_file.read().splitlines()
                #temp = image_file.read().replace('"', '').splitlines()
            #photoList = zip(*(s.split(",") for s in temp))
            photoList = temp[:-1]
            plot_id = temp[-1]
            #select the folder of the parcel in the archive map
            customer_id = os.path.basename(os.path.dirname(os.path.dirname(photoList[0])))
            day_of_recording = os.path.basename(os.path.dirname(photoList[0]))
            metashape_processing_folder = os.path.join(temp_processing_folder, customer_id, plot_id)
            ortho_out = os.path.join(processing_folder, customer_id + '-'+plot_id+'-'+day_of_recording+'.tif')
            logging.info("Customer = " + str(customer_id))
            logging.info("Day of recording = " + str(day_of_recording))
            logging.info("Plot name = " + str(plot_id))
            try:
                #register start time of metashape process
                tic = time.clock()
                #run metashape process
                timestr_save, scan_information = MetashapeProcess(photoList, day_of_recording, metashape_processing_folder, ortho_out, quality)
                #register finish time of metashape process
                toc = time.clock()
                #write processing time to log file
                processing_time = toc - tic
                logging.info("processing of " + str(len(photoList)) + " images in " + str(os.path.dirname(photoList[0])) + " finished in " + str(int(processing_time)) + " seconds \n")
                #after succesful processing move proces txt files
                shutil.move(input_file, processing_archive_path)
                nr_of_plots += 1
                nr_of_images += len(photoList)
                with open(os.path.join(os.path.dirname(photoList[0]),"processed.txt"), "w") as text_file:
                    text_file.write("Processed "+str(nr_of_plots)+ " plots and " +str(nr_of_images)+" images")
                #create df with info for processinglog in ortho inbox
                df = pd.DataFrame([[day_of_recording,timestr[:8], tic, customer_id, plot_id, nr_of_images, (str(day_of_recording) + "_" + str(timestr_save)+'.psx')]],
                    columns = ['Flight date',	'Processing date',	'Start time',	'Customer Name',	'Plot Name',	'No of photos',	'Agisoft filename'])
                #append info to processinglog.xlsx
                append_df_to_excel(os.path.join(excel_filepath, excel_filename), df)
                
                #get different paramters related to the flight/scan and add to db:
                height_of_flight, gsd, zoomlevel, flight_datetime, sensor = scan_information
                

                """
                This part of code is redundant now
                try:
                    #after metashape is succesfully finished move the images to archive
                    #output_folder for processed images (note that it differs from folder in txt file because that is the folder for metashape output)
                    timestr = time.strftime("%H%M%S")
                    output_folder_images = os.path.join(output_folder,"imagery", (str(day_of_recording)+"_"+timestr))
                    if not os.path.exists(output_folder_images):
                        os.makedirs(output_folder_images)
                    copy_files(photoList, output_folder_images)
                    #move_files_after_processing(photoList, output_folder_images)
                except Exception:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    logging.info("Error encountered at the following time: " + str(timestr))
                    logging.exception("problem with (re)moving")
                    logging.info("\n")
                """
            except Exception:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                logging.info("Error encountered at the following time: " + str(timestr))
                logging.info("Error in processing " + str(os.path.dirname(ortho_out)))
                logging.exception("Metashape processing encountered the following problem:")
                logging.info("\n")
        except Exception:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            logging.info("Error encountered at the following time: " + str(timestr))
            logging.exception("something went wrong reading processing file:")
            logging.info("\n")

"""
#after metashape processing is finished move images to archive
try:
    for proces_file in os.listdir(move_path):
        if proces_file.endswith('.txt'):
            input_file = os.path.join(move_path, proces_file)
            #with open(input_file) as image_file:
                #temp = image_file.read().replace('"', '').splitlines()
            #photoList, output = zip(*(s.split(",") for s in temp))
            #output_folder_failed_images = os.path.dirname(output[0])
            #if not os.path.exists(output_folder_failed_images):
                #os.makedirs(output_folder_failed_images)
            #try:
                #print('nu even niet')
                #move_files_after_processing(photoList, output_folder_failed_images)
            #after succesful moving the images, move the txt files to Archive
            #except:
                #timestr = time.strftime("%Y%m%d-%H%M%S")
                #logging.info("Error encountered at the following time: " + str(timestr))
                #logging.exception("files have been moved to archive")
                #logging.info("\n")
            shutil.move(input_file, processing_archive_path)
except Exception:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.info("Error encountered at the following time: " + str(timestr))
    logging.exception("problem with (re)moving failed_imagery")
    logging.info("\n")
"""
