
# -*- coding: utf-8 -*-
"""
@author: Kaz
"""
import os
from pathlib import Path
import time
import sys
#import gdal2tiles
import shutil
import gdal
import pandas as pd
import subprocess
import paramiko
from clip_ortho_2_plot import clip_ortho2plot
import logging
from vanbovendatabase.postgres_lib import *
import datetime



## FUNCTIONS ##

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

def cmd_and_wait(ssh,command):
    #Function cmd_and_wait opens an ssh channel and runs a command (command). It then
    # waits until it receives 'eof' before it returns. Without this waiting function
    #the channel would remain open until sftp.close() is called, which may not be long
    #enough to run the entire command.
    sleeptime = 0
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout.flush()
    nbytes = 0

    while not stdout.channel.eof_received:
        time.sleep(1)
        sleeptime += 1
        if sleeptime > 600:
            stdout.channel.close()
            logging.info('Broke out of cmd_and_wait with command {}'.format(command))
            break
    #if stdout.channel.eof_received:
    logging.info('Reached end of cmd_and_wait')
    #logging.info(stdout.read())
    return stdout.channel.eof_received, stdout



## CONFIG SECTION ##
inbox = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\ready' #folder where all orthos are stored
ortho_archive_destination = r'E:\VanBovenDrive\VanBoven MT\Archive' #Folder where orthos are archived (gdrive)
pem_path= r"C:\Users\VanBoven\Documents\SSH\VanBovenAdmin.pem"


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
logdir = inbox.rsplit('/',1)[0] #get parent of 'ready' folder, which is the Inbox
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
files = [file for file in os.listdir(inbox) if file.endswith('.tif')]
logging.info('Total of {} files to process: \n'.format(len(files)))
for file in files:
    #name convention: customer-plot-date.tif
    tif_count +=1
    this_customer_name,this_plot_name,this_date = file.split('-')
    this_date = this_date.split('.')[0]
    dict = {"customer_name": this_customer_name, "plot_name":this_plot_name, "date":this_date, "filename":file}
    ('    {}: {}\n'.format(str(tif_count),file))
    orthos.append(dict)

ortho_que = sorted(orthos, key=lambda k: k['date'])

## Tiling process ##

#create temporary folder for tile files if not yet available
tile_output_base_dir = os.path.join(inbox,'temp_tiles')
if not(os.path.isdir(tile_output_base_dir)):
    os.mkdir(tile_output_base_dir)
#Loop trough all available tifs and tile them.

for ortho in ortho_que:
    start_ortho_time = time.time()
    print('Started with {} at {}'.format(ortho,datetime.datetime.now().strftime('%H:%M:%S')))

    #Read dict
    customer_name = ortho['customer_name']
    plot_name = ortho['plot_name']
    date = ortho['date']
    filename = ortho['filename']
    logging.info('Starting processing of {}\n'.format(filename))

    #Check for possible duplicate work:
    ortho_archive_target = os.path.join(ortho_archive_destination,customer_name,plot_name,date,'Orthomosaic')
    if os.path.isfile(ortho_archive_target): #This needs to be tested for the 'break'
        #File already exists, quiting early and logging duplicate
        logging.info('    Ortho is already present in archive. Exited after {} seconds\n'\
        .format(round(time.time() - start_ortho_time)))
        break

    #clip ortho to plot shape:
    logging.info('    Clipping {}...\n'.format(filename))
    start_clip_time = time.time()
    clip_ortho2plot(plot_name, con, meta, inbox,filename)
    end_clip_time = time.time()
    clip_duration = round((end_clip_time-start_clip_time)/60)
    logging.info('    Clipped in {} minutes...\n'.format(clip_duration))
    filename_clipped = filename.split('.')[0]+'_clipped.'+filename.split('.')[-1]

    #Start tiling
    logging.info('Start tiling proces for {}\n'.format(filename))
    start_tiling_time = time.time()
    #Identify and create file locations
    input_file = os.path.join(inbox,filename_clipped)
    output_folder = os.path.join(tile_output_base_dir,filename.split('.')[0])
    #DEBUGGIN: Skip if already tiled
    if os.path.isdir(output_folder):
        newtile = False
    else:
        newtile = True
    os.mkdir(output_folder)
    batcmd ='python gdal2tiles.py' + ' "' + str(input_file) + '"' + ' "' + str(output_folder) + '"'+ ' -z 16-23 -w none --processes 16'
    batcmd ='python gdal2tilesroblabs.py' + ' "' + str(input_file) + '"' + ' "' + str(output_folder) + '"'+ ' -z 16-23 -w none -o tms'
    #Would be great if we can use the direct python funciton. This requires either building an options args element, or manually fixing gdal2tiles.py
    #argv= gdal.GeneralCmdLineProcessor(['"'+str(input_file)+'"','"'+str(output_folder)+'"','--processes' ,'14' ,'-z', '16-23', '-w' ,'none'])
    #input_file, output_folder, options = process_args(argv[1:])
    #multi_threaded_tiling(input_file, output_folder, gdal2tiles_options)

    if newtile:
        os.system(batcmd)

    end_tiling_time = time.time()

    logging.info('    Tiled in {} minutes \n'.format(round((end_tiling_time-start_tiling_time)/60)))

    #Zip ortho folders
    #provide make_archive with the location of the just created tile folder
    logging.info('    Zipping tiles to {}...\n'.format(output_folder))
    local_zipfile = shutil.make_archive(output_folder, 'zip', output_folder)

    #Clean up by removing the original folder. rmtree permanently removes them - be warned.
    logging.info('    Removing original tiles \n')
    shutil.rmtree(output_folder)

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
    logging.info('starting to upload',local_zipfile)
    remote_attr = sftp.put(local_zipfile,full_remote_zip_path)
    filesize = os.path.getsize(local_zipfile)
    end_upload_time = time.time()
    up_speed = filesize/(end_upload_time - start_upload_time))/1000000


    if remote_attr.st_size == filesize:
        msg = '    Succesfull upload: {}/{} MB in {} minutes at {} Mb/s.\n'.\
        format(remote_attr.st_size/1e6, filesize/1e6,round((end_upload_time-end_upload_time)/60),up_speed)
        logging.info(msg)
        os.remove(local_zipfile)
        upload_success = True
    else:
        msg = '    Failed upload: {}/{} MB in {} minutes at {} Mb/s.\n'.\
        format(remote_attr.st_size/1e6, filesize/1e6,round((end_upload_time-end_upload_time)/60),up_speed)
        logging.info(msg)
        upload_success = False


    #unzip file to the right dir and remove archive
    remote_unzip_location = '/home/ubuntu/media/data/' + customer_name + '/' + plot_name +'/' +  date
    #Specify bash ssh commands
    command_unzip = 'unzip ' + "'"+ full_remote_zip_path + "'" + ' -d ' + "'" + remote_unzip_location + "'"
    command_removezip = 'rm ' + "'"+ full_remote_zip_path + "'"

    logging.info('unzipping ',command_unzip)
    #Create folders and execute commands
    mkpath(sftp,remote_unzip_location)


    logging.info('    Unzipping...\n')
    cmd_and_wait(ssh,command_unzip)
    logging.info('deleting ',remote_unzip_location)
    #delete zip file from webserver
    logging.info('    Cleaning up...\n')
    cmd_and_wait(ssh,command_removezip)
    sftp.close()
    logging.info('moving files')
    #Move orthomosaic to correct folder
    if not(os.path.isdir(ortho_archive_target)):
        os.makedirs(ortho_archive_target)

    #Moving original and clipped ortho to archive
    shutil.move(input_file,os.path.join(ortho_archive_target,filename_clipped))
    shutil.move(os.path.join(inbox,filename),os.path.join(ortho_archive_target,filename))


    end_ortho_time = time.time()
    logging.info('    Finished processing {} in {} minutes\n \n'.\
    format(filename,round((start_ortho_time-end_ortho_time)/60)))

    insert_new_scan(date, plot_name, meta, con)
    logging.info("Added scan {} to plot {}".format(date,plot_name))


con.connect().close()
print('Reached end of script')
logging.info('Reached end of script')
