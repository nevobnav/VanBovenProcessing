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
#1. Load images
def getPhotoList(path, photoList):
    pattern = '.JPG$'
    #for root, dirs, files in os.walk(path):
    for name in os.listdir(path):
        if re.search(pattern,name):
            cur_path = os.path.join(path, name)
            #print (cur_path)
            photoList.append(cur_path)
"""
import Metashape
import os,re,sys
import time
import logging
import shutil

def move_files_after_processing(photoList, output_folder):
    for photo in photoList:
        shutil.move(photo, output_folder)
    fileList = os.listdir(os.path.dirname(photoList[0]))
    if len(fileList) > 0:
        for file in fileList:
            shutil.move(os.path.join(os.path.dirname(photoList[0]), file), output_folder)
    #can give trouble with permission in windows
    os.removedirs(os.path.dirname(photoList[0]))

def getAltitude(chunk):
    for camera in chunk.cameras:
    	if "DJI/RelativeAltitude" in camera.photo.meta.keys() and camera.reference.location:
    		z = float(camera.photo.meta["DJI/RelativeAltitude"])
    		camera.reference.location = (camera.reference.location.x, camera.reference.location.y, z)
    chunk.updateTransform()
    Metashape.app.update()
    print("Script finished")

def MetashapeProcess(photoList, output_folder, day_of_recording):
    #if folder.endswith('Perceel1'): ook een optie afhankelijk van naamgeving mappen
    #path = folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #start with cleared console
    Metashape.app.console.clear()

    ## construct the document class
    doc = Metashape.app.document

    ## save project
    #doc.open("M:/Metashape/practise.psx")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    psxfile = output_folder + '\\' + str(timestr)+'.psx'
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
    ################################################################################################
    ### align photos ###
    ## Perform image matching for the chunk frame.
    # - Alignment accuracy in [HighestAccuracy, HighAccuracy, MediumAccuracy, LowAccuracy, LowestAccuracy]
    # - Image pair preselection in [ReferencePreselection, GenericPreselection, NoPreselection]
    chunk.matchPhotos(accuracy=Metashape.HighestAccuracy, preselection=Metashape.ReferencePreselection, filter_mask=False, keypoint_limit=40000, tiepoint_limit=4000)
    chunk.alignCameras()

    #optional incteratively increase camera accuracy
    threshold = 0.5
    f = Metashape.PointCloud.Filter()
    f.init(chunk, criterion = Metashape.PointCloud.Filter.ReprojectionError)
    f.removePoints(threshold)
    #optimize cameras based on accurate points
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True, fit_k1=True,
    fit_k2=True, fit_k3=True, fit_k4=False, fit_p1=True, fit_p2=True, fit_p3=False,
    fit_p4=False, adaptive_fitting=False, tiepoint_covariance=False)

    ################################################################################################
    ### build dense cloud ###
    ## Generate depth maps for the chunk.
    # buildDenseCloud(quality=MediumQuality, filter=AggressiveFiltering[, cameras], keep_depth=False, reuse_depth=False[, progress])
    # - Dense point cloud quality in [UltraQuality, HighQuality, MediumQuality, LowQuality, LowestQuality]
    #every step lower dan UltraQuality downscales the images by a factor 4 (2x per side)
    # - Depth filtering mode in [AggressiveFiltering, ModerateFiltering, MildFiltering, NoFiltering]
    chunk.buildDepthMaps(quality=Metashape.LowQuality, filter=Metashape.MildFiltering)
    chunk.buildDenseCloud(max_neighbors = 100, point_colors = False)

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
    doc.save()

    ################################################################################################
    ### build DEM (before build dem, you need to save the project into psx) ###
    ## Build elevation model for the chunk.
    # buildDem(source=DenseCloudData, interpolation=EnabledInterpolation[, projection ][, region ][, classes][, progress])
    # - Data source in [PointCloudData, DenseCloudData, ModelData, ElevationData]
    chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation, projection=chunk.crs)

    ################################################################################################
    ## Build orthomosaic for the chunk.
    # buildOrthomosaic(surface=ElevationData, blending=MosaicBlending, color_correction=False[, projection ][, region ][, dx ][, dy ][, progress])
    # - Data source in [PointCloudData, DenseCloudData, ModelData, ElevationData]
    # - Blending mode in [AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending]
    chunk.buildOrthomosaic(surface=Metashape.ElevationData, blending=Metashape.MosaicBlending, projection=chunk.crs)

    ################################################################################################
    ## auto classify ground points (optional)
    #chunk.dense_cloud.classifyGroundPoints()
    #chunk.buildDem(source=Metashape.DenseCloudData, classes=[2])

    ################################################################################################
    doc.save()

    if not os.path.exists(output_folder+"\\Orthomosaic\\"):
        os.makedirs(output_folder+"\\Orthomosaic\\")

    #zorg voor mooie naamgeving + output
    chunk.exportOrthomosaic(path = output_folder+"\\Orthomosaic\\" + day_of_recording + '.tif')

#Start of execution

#initiate log file
timestr = time.strftime("%Y%m%d-%H%M%S")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = r"E:\VanBovenDrive\VanBoven MT\Processing\Log_files/Metashape_log_file_" +str(timestr) +  ".log",level=logging.DEBUG)

#root_path = r'E:\100 Testing\190304 batch_script_test'
process_path = r'E:\VanBovenDrive\VanBoven MT\Processing\To_process'
move_path = r'E:\VanBovenDrive\VanBoven MT\Processing\To_move'
processing_archive_path = r'E:\VanBovenDrive\VanBoven MT\Processing\Archive'
#execute:
try:
    #iterate through the folder with processing txt files
    for proces_file in os.listdir(process_path):
        if proces_file.endswith('.txt'):
            input_file = os.path.join(process_path, proces_file)
            with open(input_file) as image_file:
                temp = image_file.read().replace('"', '').splitlines()
            photoList, output = zip(*(s.split(",") for s in temp))
            #select the folder of the parcel in the archive map
            customer_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(output[0]))))
            plot_id = os.path.basename(os.path.dirname(os.path.dirname(output[0])))
            day_of_recording = os.path.basename(os.path.dirname(output[0]))
            output_folder = os.path.dirname(os.path.dirname(output[0]))
            try:
                #register start time of metashape process
                tic = time.clock()
                #run metashape process
                MetashapeProcess(photoList, output_folder, day_of_recording)
                #register finish time of metashape process
                toc = time.clock()
                #write processing time to log file
                processing_time = toc - tic
                logging.info("processing of " + str(len(photoList)) + " images in " + str(os.path.dirname(output[0])) + " finished in " + str(processing_time) + " seconds I guess")
                #after succesful processing move proces txt files
                shutil.move(input_file, processing_archive_path)
                try:
                    #after metashape is succesfully finished move the images to archive
                    #output_folder for processed images (note that it differs from folder in txt file because that is the folder for metashape output)
                    output_folder_images = os.path.join(output_folder,"imagery", str(day_of_recording))
                    if not os.path.exists(output_folder_images):
                        os.makedirs(output_folder_images)
                    move_files_after_processing(photoList, output_folder_images)
                except Exception:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    logging.info("Error encountered at the following time: " + str(timestr))
                    logging.exception("problem with (re)moving")
                    logging.info("\n")
            except Exception:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                logging.info("Error encountered at the following time: " + str(timestr))
                logging.info("Error in processing " + str(os.path.dirname(output[0])))
                logging.exception("Metashape processing encountered the following problem:")
                logging.info("\n")
except Exception:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.info("Error encountered at the following time: " + str(timestr))
    logging.exception("something went wrong reading processing file:")
    logging.info("\n")

#after metashape processing is finished move images that were not selected for processing to archive
try:
    for proces_file in os.listdir(move_path):
        if proces_file.endswith('.txt'):
            input_file = os.path.join(move_path, proces_file)
            with open(input_file) as image_file:
                temp = image_file.read().replace('"', '').splitlines()
            photoList, output = zip(*(s.split(",") for s in temp))
            output_folder_failed_images = os.path.dirname(output[0])
            if not os.path.exists(output_folder_failed_images):
                os.makedirs(output_folder_failed_images)
            move_files_after_processing(photoList, output_folder_failed_images)
            #after succesful moving the images, move the txt files to Archive
            shutil.move(input_file, processing_archive_path)
except Exception:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.info("Error encountered at the following time: " + str(timestr))
    logging.exception("problem with (re)moving failed_imagery")
    logging.info("\n")