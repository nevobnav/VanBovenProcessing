
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
import select



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
    remote_unzip_location = '/home/ubuntu/media/data/' + customer_name + '/' + plot_name +'/' +  date
    #Specify bash ssh commands
    command_unzip = 'unzip ' + "'"+ full_remote_zip_path + "'" + ' -d ' + "'" + remote_unzip_location + "'"
    command_removezip = 'rm ' + "'"+ full_remote_zip_path + "'"

    logging.info('unzipping {}'.format(command_unzip))
    #Create folders and execute commands
    mkpath(sftp,remote_unzip_location)


    start_zip_time = time.time()

    zip_output = exec_ssh(ssh, command_unzip)
    end_zip_time = time.time()
    logging.info('    Unzipped in {} seconds. \n'.format(end_zip_time - start_zip_time))

    logging.info('deleting {}. \n'.format(remote_unzip_location))
    #delete zip file from webserver
    start_delete_time = time.time()
    exec_ssh(ssh,command_removezip)
    end_delete_time = time.time()
    logging.info('deleted in {} seconds. \n'.format(end_delete_time - start_delete_time))
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
