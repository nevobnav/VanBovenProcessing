
# -*- coding: utf-8 -*-
"""
@author: Kaz
"""
import os
import sys
import time
import shutil

import paramiko
#from clip_ortho_2_plot import clip_ortho2plot
from clip_ortho_2_plot_gdal import clip_ortho2plot_gdal

import logging
from vanbovendatabase.postgres_lib import *
import datetime
import select
import multiprocessing
import gdal2tiles

#23 is default
zoomlevel = 23
intensity_flag = sys.argv[1]

## FUNCTIONS ##

def exec_ssh(ssh, cmd, timeout=1200, want_exitcode=False):
    #source: https://stackoverflow.com/questions/23504126/do-you-have-to-check-exit-status-ready-if-you-are-going-to-check-recv-ready/32758464#32758464
    # one channel per command
    stdin, stdout, stderr = ssh.exec_command(cmd)
    # get the shared channel for stdout/stderr/stdin
    channel = stdout.channel
    # we do not need stdin.
    stdin.close()
    # indicate that we're not going to write to that channel anymore
    channel.shutdown_write()

    # read stdout/stderr in order to prevent read block hangs
    stdout_chunks = []
    stdout_chunks.append(stdout.channel.recv(len(stdout.channel.in_buffer)))
    # chunked read to prevent stalls
    while not channel.closed or channel.recv_ready() or channel.recv_stderr_ready():
      # stop if channel was closed prematurely, and there is no data in the buffers.
      got_chunk = False
      readq, _, _ = select.select([stdout.channel], [], [], timeout)
      for c in readq:
          if c.recv_ready():
              stdout_chunks.append(stdout.channel.recv(len(c.in_buffer)))
              got_chunk = True
          if c.recv_stderr_ready():
              # make sure to read stderr to prevent stall
              stderr.channel.recv_stderr(len(c.in_stderr_buffer))
              got_chunk = True
      '''
      1) make sure that there are at least 2 cycles with no data in the input buffers in order to not exit too early (i.e. cat on a >200k file).
      2) if no data arrived in the last loop, check if we already received the exit code
      3) check if input buffers are empty
      4) exit the loop
      '''
      if not got_chunk \
          and stdout.channel.exit_status_ready() \
          and not stderr.channel.recv_stderr_ready() \
          and not stdout.channel.recv_ready():
          # indicate that we're not going to read from this channel anymore
          stdout.channel.shutdown_read()
          # close the channel
          stdout.channel.close()
          break    # exit as remote side is finished and our bufferes are empty

    # close all the pseudofiles
    stdout.close()
    stderr.close()

    if want_exitcode:
        #exit code is always ready at this point
        return (stdout.channel.recv_exit_status())
    return stdout_chunks



def mkpath(sftp,path):
    #Function mkpath takes in sftp object and a desired path. The function starts
    #testing if parent directory exists. If it doesn't, it moves up the chain until
    #a parent exists. At that point it creates
    try:
        sftp.chdir(path)
        return
    except IOError:
        parent = path.rsplit('/',1)[0] #get parent directory
        #test if parent exists
        if not parent:
            return
        try:
            sftp.chdir(parent)
        except IOError:
            mkpath(sftp,parent)
        sftp.mkdir(path)


## CONFIG SECTION ##
pem_path= r"C:\Users\VanBoven\Documents\SSH\VanBovenAdmin.pem"

if multiprocessing.cpu_count() < 32:
    path_ready_to_rectify = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\1_ready_to_rectify'       # folder where all approved original orthos, DEMS and points are stored
    path_rectified_DEMs = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\00_rectified_DEMs_points'   # folder where all rectified DEMs are stored
    path_ready_to_upload = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\2_ready_to_upload'         # folder where all rectified orthos are stored

    path_trashbin_originals = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\00_trashbin_originals'  # temporary folder where original orthos and DEMs are kept AFTER georectification
elif multiprocessing.cpu_count() >= 32:
    path_ready_to_rectify = r'd:\100 Ortho Inbox\1_ready_to_rectify'       # folder where all approved original orthos, DEMS and points are stored
    path_rectified_DEMs = r'd:\100 Ortho Inbox\00_rectified_DEMs_points'   # folder where all rectified DEMs are stored
    path_ready_to_upload = r'd:\100 Ortho Inbox\2_ready_to_upload'         # folder where all rectified orthos are stored

    path_trashbin_originals = r'd:\100 Ortho Inbox\00_trashbin_originals'  # temporary folder where original orthos and DEMs are kept AFTER georectification

ortho_archive_destination = r"A:\\"                             # Folder where rectified orthos, DEMs and points are archived (gdrive)

with open('postgis_config.json') as config_file:
    config = json.load(config_file)
DB_NAME = config['NAME']
DB_USER = config['DB_USER']
DB_PASSWORD = config['DB_PASSWORD']
DB_IP = config['DB_IP']

con,meta = connect(DB_USER, DB_PASSWORD, DB_NAME, host=DB_IP)
#initialize
orthos = []
tif_count = 0
duplicate = 0
timestr = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
datestr = datetime.datetime.now().strftime('%Y%m%d')
timestr_filename = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

## Creating log file:
logdir = os.path.join(path_ready_to_upload,'logs') #get parent of 'ready' folder, which is the Inbox
logfile = '{}_log.txt'.format(timestr_filename)
logpath_base = os.path.join(logdir,logfile)
logpath = logpath_base
while os.path.isfile(logpath):
    logpath = logpath_base[:-4]+'('+str(duplicate)+')'+".txt"
    duplicate += 1

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = logpath,level=logging.INFO)

logging.info('Time: {}\n'.format(timestr))


#Collect available tif files and list of dicts 'orthos'
files = [file for file in os.listdir(path_ready_to_upload) if (file.endswith('.tif')) or (file.endswith('.vrt')) and not (file.endswith('_DEM.tif'))]
logging.info('Total of {} files to process: \n'.format(len(files)))
for file in files:

    # 2 scenario's for naming convention:
        # 1) not rectified: only customer-plot-date.tif exists
        # 2) rectified: customer-plot-date.tif AND customer-plot-date-GR.vrt exists    tif_count +=1

    test = file.split('-')


    if len(test) == 3:
        this_customer_name,this_plot_name,this_datetime = file.split('-')
    elif len(test) == 4:
        this_customer_name,this_plot_name,this_datetime, this_GR = file.split('-')

    this_datetime = this_datetime.split('.')[0]
    this_datetime = this_datetime.split('(')[0] #Splits of brackets for duplicates if present
    this_datetime = this_datetime.replace('_clipped','') #removes the _clipped addition if present
    this_date = this_datetime[0:8]
    this_time = this_datetime[8:13]

    #Fix cases where no time is available in the file name:
    if len(this_time) == 0:
        this_time = '0200'

    if len(test) == 4:
        dict = {"customer_name": this_customer_name, "plot_name":this_plot_name, "flight_date":this_date, "flight_time":this_time, "filename":file, "georectified": True}
        ('    {}: {}\n'.format(str(tif_count),file))
        orthos.append(dict)

    # logging.info(str(len(test)))
    # logging.info(str(file))
    # logging.info(str(this_customer_name))
    # logging.info(str(this_plot_name))
    # logging.info(str(this_date))
    # logging.info(str(this_time))
    # logging.info(str(file))

    if len(test) == 3:
    #elif len(test) == 3:
        # if a tiff not rectified, the VRT should NOT exist
        if file.endswith('.tif'):
            checkname = file.replace('.tif', '-GR.vrt')
            if checkname not in files:
                dict = {"customer_name": this_customer_name, "plot_name":this_plot_name, "flight_date":this_date, "flight_time":this_time, "filename":file, "georectified": False}
                ('    {}: {}\n'.format(str(tif_count),file))
                orthos.append(dict)

ortho_que = sorted(orthos, key=lambda k: k['flight_date'])
logging.info(len(ortho_que))
print(ortho_que)
logging.info(len(orthos))


## Tiling process ##

#create temporary folder for tile files if not yet available
tile_output_base_dir = os.path.join(path_ready_to_upload,'temp_tiles')
if not(os.path.isdir(tile_output_base_dir)):
    os.mkdir(tile_output_base_dir)

#Loop trough all available tifs and tile them.
for ortho in ortho_que:
    start_ortho_time = time.time()
    print('Started with {} at {}'.format(ortho,datetime.datetime.now().strftime('%H:%M:%S')))

    #Read dict
    customer_name = ortho['customer_name']
    plot_name = ortho['plot_name']
    flight_date = ortho['flight_date']
    flight_time = ortho['flight_time']
    filename = ortho['filename']
    logging.info('Starting processing of {}\n'.format(filename))
    georectified = ortho['georectified']

    #Fetch zoomlevel from database entry (zoomlevel established in Batch_Processing)
    scan_data = get_scan(con, meta, date=flight_date, time=flight_time, plot=plot_name)
    logging.info('      Found scan details: {}'.format(str(scan_data)))
    try:
        scan_id, zoomlevel = scan_data[0]['scan_id'], scan_data[0]['zoomlevel']
    except:
        logging.info('      No database entry with date {}, time {} and plot {}. Quiting.'.format(flight_date, flight_time, plot_name))
        pass


    #Check for possible duplicate work:
    ortho_archive_target = os.path.join(ortho_archive_destination,customer_name,plot_name,flight_date,flight_time,'Orthomosaic')
    if os.path.isfile(ortho_archive_target): #This needs to be tested for the 'break'
        #File already exists, quiting early and logging duplicate
        logging.info('    Ortho is already present in archive. Exited after {} seconds\n'\
        .format(round(time.time() - start_ortho_time)))


    #clip ortho to plot shape, if clipped file isnt already in the folder:
    logging.info('    Clipping {}...\n'.format(filename))
    filename_clipped = filename.split('.')[0]+'_clipped.tif'

    start_clip_time = time.time()

    if not(os.path.exists(os.path.join(path_ready_to_upload,filename_clipped))):
        clip_ortho2plot_gdal(plot_name, con, meta, path_ready_to_upload,filename)

    end_clip_time = time.time()
    clip_duration = round((end_clip_time-start_clip_time)/60)
    logging.info('    Clipped in {} minutes...\n'.format(clip_duration))


    #Start tiling
    logging.info('Start tiling proces for {}\n'.format(filename))
    start_tiling_time = time.time()

    #Identify and create file locations
    input_file = os.path.join(path_ready_to_upload,filename_clipped)
    output_folder = os.path.join(tile_output_base_dir,filename.split('.')[0])

    #DEBUGGIN: Skip if already tiled
    if os.path.exists(output_folder+'.zip'):
        newtile = False
        logging.info('   Tiles already present, skip tiling')
    else:
        newtile = True
        os.mkdir(output_folder)

    # make bat string for parallel gdal2tiles
    if intensity_flag == 'low':
        batcmd ='python gdal2tilesroblabs.py' + ' "' + str(input_file) + '"' + ' "' + str(output_folder) + '"'+ ' -z 16-'+ str(zoomlevel) +' -w none -o tms' +' --processes 14'
    else:
        batcmd ='python gdal2tilesroblabs.py' + ' "' + str(input_file) + '"' + ' "' + str(output_folder) + '"'+ ' -z 16-'+ str(zoomlevel) +' -w none -o tms' +' --processes 2'

    if newtile:
        os.system(batcmd)
#        # gdal2tiles using python bindings
#        #no_of_cpus = multiprocessing.cpu_count()
#        #tiling_options = {'zoom': [16, zoomlevel], 'tmscompatible': True, 'nb_processes':int(no_of_cpus/2-1), 'webviewer': 'none'}
#        tiling_options = {'zoom': [16, zoomlevel], 'tmscompatible': True, 'nb_processes':1, 'webviewer': 'none'}
#        try:
#            gdal2tiles.generate_tiles(input_file, output_folder, **tiling_options)
#            continue
#        except:
#            print('Tile generation did not work \n')
#            continue

        end_tiling_time = time.time()

        logging.info('    Tiled in {} minutes \n'.format(round((end_tiling_time-start_tiling_time)/60)))

        #Zip ortho folders
        #provide make_archive with the location of the just created tile folder
        logging.info('    Zipping tiles to {}...\n'.format(output_folder))
        local_zipfile = shutil.make_archive(output_folder, 'zip', output_folder)

        #Clean up by removing the original folder. rmtree permanently removes them - be warned.
        logging.info('    Removing original tiles \n')
        shutil.rmtree(output_folder)

    else:
        local_zipfile = output_folder + '.zip'

    ## Uploading ##
    logging.info('    Preparing to upload.\n')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ec2-52-29-220-114.eu-central-1.compute.amazonaws.com',
                username='ubuntu',
                key_filename=pem_path)
    sftp = ssh.open_sftp()

    #create the right dir:
    remote_base = '/home/ubuntu/media/data/zipfiles'

    #upload zip to remote_dir
    full_remote_zip_path = remote_base + '/' + filename.split('.')[0] + '.zip'
    start_upload_time = time.time()
    logging.info('starting to upload {}'.format(local_zipfile))
    remote_attr = sftp.put(local_zipfile,full_remote_zip_path)
    filesize = os.path.getsize(local_zipfile)
    end_upload_time = time.time()
    up_speed = filesize/(end_upload_time - start_upload_time)/1000000


    if remote_attr.st_size == filesize:
        msg = '    Succesfull upload: {}/{} MB in {} minutes at {} Mb/s.\n'.\
        format(remote_attr.st_size/1e6, filesize/1e6,round((end_upload_time-start_upload_time)/60),up_speed)
        logging.info(msg)
        os.remove(local_zipfile)
        upload_success = True
    else:
        msg = '    Failed upload: {}/{} MB in {} minutes at {} Mb/s.\n'.\
        format(remote_attr.st_size/1e6, filesize/1e6,round((end_upload_time-start_upload_time)/60),up_speed)
        logging.info(msg)
        upload_success = False


    #unzip file to the right dir and remove archive

    remote_unzip_location = '/home/ubuntu/media/data/' + customer_name + '/' + plot_name +'/' +  flight_date + flight_time
    #Specify bash ssh commands
    command_unzip = 'unzip ' + "'"+ full_remote_zip_path + "'" + ' -d ' + "'" + remote_unzip_location + "'"
    command_removezip = 'rm ' + "'"+ full_remote_zip_path + "'"

    logging.info('unzipping {}'.format(command_unzip))

    #Clear out folder in case old tiles are present
    try:
        command_remove_folder = 'rm -r ' + "'" + remote_unzip_location + "'"
        logging.info('clearing out {}'.format(command_remove_folder))
        remove_folder_output = exec_ssh(ssh, command_remove_folder)
        logging.info('Removed folder {}'.format(str(remove_folder_output)))
    except Exception as e:
        logging.info('Error while removing image folder: {}'.format(e))

    #Create folders and execute commands
    mkpath(sftp,remote_unzip_location)

    start_zip_time = time.time()

    zip_output = exec_ssh(ssh, command_unzip)
    end_zip_time = time.time()
    logging.info('    Unzipped in {} seconds. \n'.format(end_zip_time - start_zip_time))

    logging.info('deleting {}. \n'.format(full_remote_zip_path))
    #delete zip file from webserver
    start_delete_time = time.time()
    exec_ssh(ssh,command_removezip)
    end_delete_time = time.time()
    logging.info('deleted in {} seconds. \n'.format(end_delete_time - start_delete_time))
    sftp.close()
    logging.info('moving and removing files')

    ## Move rectified ortho, DEM and .points file to archive

    if georectified == False:
        filename_ortho_or = filename
        filename_DEM_or = os.path.splitext(filename_ortho_or)[0] + '_DEM.tif'

    #define all filenames
    else:
        filename_ortho_vrt = filename
        filename_ortho_or = os.path.splitext(filename)[0][:-3]+'.tif'
        filename_DEM_vrt = os.path.splitext(filename)[0][:-3]+'_DEM-GR.vrt'
        filename_DEM_or = os.path.splitext(filename)[0][:-3]+'_DEM.tif'
        filename_points = os.path.splitext(filename_ortho_or)[0]+'.points'

    #Moving (georectified) ortho to archive
    if not(os.path.isdir(ortho_archive_target)):
        os.makedirs(ortho_archive_target)

    #Move ortho and DEM to archive
    #Move both original files + vrt files and points file to archive if files are georectified
    if georectified == True:
        shutil.move(os.path.join(path_ready_to_upload, filename_ortho_or),os.path.join(ortho_archive_target, filename_ortho_or))
        shutil.move(os.path.join(path_ready_to_upload, filename_ortho_vrt),os.path.join(ortho_archive_target, filename_ortho_vrt))
        shutil.move(os.path.join(path_rectified_DEMs, filename_DEM_or),os.path.join(ortho_archive_target, filename_DEM_or))
        shutil.move(os.path.join(path_rectified_DEMs, filename_DEM_vrt),os.path.join(ortho_archive_target, filename_DEM_vrt))
        shutil.move(os.path.join(path_rectified_DEMs, filename_points), os.path.join(ortho_archive_target, filename_points))
    #Move only original files to archive if images have not been georectified
    elif georectified == False:
        shutil.move(os.path.join(path_ready_to_upload, filename_ortho_or),os.path.join(ortho_archive_target, filename_ortho_or))
        if os.path.exists(os.path.join(path_rectified_DEMs, filename_DEM_or)):
            shutil.move(os.path.join(path_rectified_DEMs, filename_DEM_or),os.path.join(ortho_archive_target, filename_DEM_or))
        else:
            logging.info("Original DEM of " + filename_ortho_or + " doesn't exist")

    # Clear out trashbin_originals as all files have been processed, only in case of rectified stuff
    # if ortho['georectified']:
    #     try:
    #         os.remove(os.path.join(path_trashbin_originals, filename_ortho_or))
    #     except:
    #         print('removing original ortho from trashbin did not happen')
    #
    #     try:
    #         os.remove(os.path.join(path_trashbin_originals, filename_DEM_or))
    #     except:
    #         print('removing original DEM from trashbin did not happen')

    # Remove clipped ortho
    os.remove(os.path.join(path_ready_to_upload, filename_clipped))

    end_ortho_time = time.time()
    logging.info('    Finished processing {} in {} minutes\n \n'.\
    format(filename,round((start_ortho_time-end_ortho_time)/60)))

    scan_time = datetime.datetime.strptime(flight_time, '%H%M').time()

    #Update scan database with tiling time and being live
    update_dict = {'live':True,'tiles':datetime.datetime.now()}
    update_scan(con,meta,update_dict,scan_id = scan_id)
    logging.info("Updated scan {}-{} of plot {}".format(flight_date,flight_time,plot_name))

con.dispose()
print('Reached end of script')
logging.info('Reached end of script')
