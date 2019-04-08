# -*- coding: utf-8 -*-
"""
@author: Kaz
"""
import os
from pathlib import Path
import time
import datetime
import sys
#import gdal2tiles
import shutil
import pandas as pd
import subprocess
import paramiko
from gdal2tilesp import gdal2tilesp
import logging


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
        print(sleeptime)
        if sleeptime > 300:
            stdout.channel.close()
            break
    if stdout.channel.eof_received:
        print('EoF received!')
    print(stdout.read())
    return stdout.channel.eof_received, stdout



## CONFIG SECTION ##
inbox = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\ready' #folder where all orthos are stored
ortho_archive_destination = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\ready' #Folder where orthos are archived (gdrive)
pem_path= r"C:\Users\VanBoven\Documents\SSH\VanBovenAdmin.pem"
logfolder = os.path.join(inbox,'logfiles')
gdal2tilesp_location = os.path.join(Path.cwd(),'gdal2tilesp','gdal2tilesp.py') #location of gdal2tiles.py folder


#DEV LINES:
inbox = '/Users/kazv/Desktop/inbox/ready'
pem_path = '/Users/kazv/.ssh/VanBovenAdmin.pem'
ortho_archive_destination = '/Users/kazv/Desktop/ortho_archive/'


#initialize
orthos = []
tif_count = 0
duplicate = 0



## Creating log file:
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = + str(timestr) + "_process_from_inbox_log_file.log",level=logging.DEBUG)


logdir = inbox.rsplit('/',1)[0] #get parent of 'ready' folder, which is the Inbox
logfile = '{}_log.txt'.format(datetime.datetime.now().strftime('%Y%m%d'))
logpath_base = os.path.join(logdir,logfile)
logpath = logpath_base
while os.path.isfile(logpath):
    logpath = logpath_base[:-4]+'('+str(duplicate)+')'+".txt"
    duplicate += 1
f = open(logpath,"a+")

f.write('Time: {}\n'.format(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')))


#Collect available tif files and list of dicts 'orthos'
files = [file for file in os.listdir(inbox) if file.endswith('.tif')]
f.write('Total of {} files to process: \n'.format(len(files)))
for file in files:
    #name convention: customer-plot-date.tif
    tif_count +=1
    this_customer,this_plot_name,this_date = file.split('-')
    this_date = this_date.split('.')[0]
    dict = {"customer": this_customer, "plot_name":this_plot_name, "date":this_date, "filename":file}
    f.write('    {}: {}\n'.format(str(tif_count),file))
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

    #Read dict
    customer = ortho['customer']
    plot_name = ortho['plot_name']
    date = ortho['date']
    filename = ortho['filename']
    f.write('Starting processing of {}\n'.format(filename))

    #Check for possible duplicate work:
    ortho_archive_target = os.path.join(ortho_archive_destination,customer,plot_name,date,'orthomosaic')
    if os.path.isfile(ortho_archive_target):
        #File already exists, quiting early and logging duplicate
        f.write('    Ortho is already present in archive. Exited after {} seconds\n'\
        .format(round(time.time() - start_ortho_time)))
        break   #THIS GIVES AN ERRO!




    f.write('    Tiling {}...\n'.format(filename))

    #Identify and create file locations
    input_file = os.path.join(inbox,filename)
    output_file = os.path.join(tile_output_base_dir,filename.split('.')[0])

    #DEBUGGIN: Skip if already tiled
    if os.path.isdir(output_file):
        newtile = False
    else:
        newtile = True
        os.mkdir(output_file)


    #Start tiling
    print('Start tiling proces for',filename)
    start_tiling_time = time.time()
    batcmd =gdal2tilesp_location+ ' "' + str(input_file) + '"' + ' "' + str(output_file) + '"'+ ' -z 16-22 -w none -o tms'
    if newtile:
        os.system(batcmd)

    end_tiling_time = time.time()

    f.write('    Tiled in {} minutes \n'.format(round((end_tiling_time-start_tiling_time)/60)))

    #Zip ortho folders
    #provide make_archive with the location of the just created tile folder
    f.write('    Zipping tiles to {}...\n'.format(output_file))
    local_zipfile = shutil.make_archive(output_file, 'zip', output_file)

    #Clean up by removing the original folder. rmtree permanently removes them - be warned.
    f.write('    Removing original tiles \n')
    shutil.rmtree(output_file)

    ## Uploading ##
    f.write('    Preparing to upload.\n')
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
    print('starting to upload',local_zipfile)
    remote_attr = sftp.put(local_zipfile,full_remote_zip_path)
    filesize = os.path.getsize(local_zipfile)
    end_upload_time = time.time()

    if remote_attr.st_size == filesize:
        msg = '    Succesfull upload: {}/{} MB in {} minutes.\n'.\
        format(remote_attr.st_size/1e6, filesize/1e6,round((end_upload_time-end_upload_time)/60) )
        f.write(msg)
        os.remove(local_zipfile)
        upload_success = True
    else:
        msg = '    Failed upload: {}/{} MB in {} minutes.\n'.\
        format(remote_attr.st_size/1e6, filesize/1e6,round((end_upload_time-end_upload_time)/60) )
        f.write(msg)
        upload_success = False


    #unzip file to the right dir and remove archive
    remote_unzip_location = '/home/ubuntu/media/data/' + customer + '/' +plot_name +'/' +  date
    #Specify bash ssh commands
    command_unzip = 'unzip ' + "'"+ full_remote_zip_path + "'" + ' -d ' + "'" + remote_unzip_location + "'"
    command_removezip = 'rm ' + "'"+ full_remote_zip_path + "'"

    print('unzipping ',command_unzip)
    #Create folders and execute commands
    mkpath(sftp,remote_unzip_location)


    f.write('    Unzipping...\n')
    cmd_and_wait(ssh,command_unzip)
    print('deleting ',remote_unzip_location)
    #delete zip file from webserver
    f.write('    Cleaning up...\n')
    cmd_and_wait(ssh,command_removezip)
    sftp.close()
    print('moving files')
    #Move orthomosaic to correct folder
    if not(os.path.isdir(ortho_archive_target)):
        os.makedirs(ortho_archive_target)
    shutil.move(input_file,os.path.join(ortho_archive_target,filename))

    end_ortho_time = time.time()
    f.write('    Finished processing {} in {} minutes\n \n'.\
    format(filename,round((start_ortho_time-end_ortho_time)/60)))

    plot_id = plotname2id(plotname, meta,con)
    insert_new_scan(scan_id, plot_id, meta, con)
    print("Added scan {} to plot {}: {}".format(scan_id,plot_id,plotname))
