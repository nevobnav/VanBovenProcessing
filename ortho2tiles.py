# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:54:50 2019

@author: VanBoven
"""

import os
#import gdal2tiles
import pandas as pd
import subprocess

os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing\gdal2tilesp')
import gdal2tilesp

#folder where all orthos are stored
folder = r'E:\VanBovenDrive\VanBoven MT\Archive'
output_folder = r'C:\Tiles4'
skipme = ['Wever oost','Rijweg stalling 1','Rijweg stalling 2','unknown_plot_id']

#get lists of folders for processing
folderList = pd.DataFrame([x[0] for x in os.walk(folder)], columns = ['Path'])
folderList['Orthomosaic'] = folderList.Path.apply(lambda x: 'Orthomosaic' in x)
nr_of_subdirs = folderList.Path.str.split("\\")
folderList['Nr_of_subdirs'] = nr_of_subdirs.apply(lambda x:len(x))
orthoList = folderList[folderList['Orthomosaic'] == True]
tileList = orthoList.Path.apply(lambda x:x.replace('Orthomosaic', 'Tiles'))

#loop through orthoList for processing
for i, path in enumerate(orthoList['Path']):
    if orthoList.Nr_of_subdirs.iloc[i] > 7:
        plot_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    else:
        plot_name = os.path.basename(os.path.dirname(path))
    orthos = os.listdir(path)
    try:
        tiles = os.listdir(tileList.iloc[i])
    except:
        'path does not exist'
    for ortho in orthos:
        if ortho.endswith('.tif'):
            ortho_date = ortho[0:8]
            try:
                if any(ortho_date in s for s in tiles):
                    'Tiles exist allready'
                else:
                    output_path = os.path.join(output_folder, plot_name)
                    #process orthos into tiles
                    input_file = os.path.join(path, ortho)
                    output_file = os.path.join(output_path, ortho_date)
                    #gdal2tiles.generate_tiles(input_file, output_file, zoom = '16-24')
                    #multicore version
                    print('Procesing in main ELSE loop')
                    if not (plot_name in skipme):
                        batcmd = r'C:/Users/VanBoven/Documents/GitHub/VanBovenProcessing/gdal2tilesp/gdal2tilesp.py ' + '"' + str(input_file) + '"' + ' "' + str(output_file) + '"'+ ' -z 16-24 -w leaflet -o tms'
                        os.system(batcmd)
                    else:
                        print('Already did these')
            except:
                output_path = os.path.join(output_folder, plot_name)
                #process orthos into tiles
                input_file = os.path.join(path, ortho)
                output_file = os.path.join(output_path, ortho_date)
                if not (plot_name in skipme):
                    os.makedirs(os.path.join(output_path, ortho_date))
                    #gdal2tiles.generate_tiles(input_file, output_file, zoom = '16-24')
                    #multicore version
                    print('Procesing in main ELSE loop')
                    batcmd = r'C:/Users/VanBoven/Documents/GitHub/VanBovenProcessing/gdal2tilesp/gdal2tilesp.py ' + '"' + str(input_file) + '"' + ' "' + str(output_file) +'"'+ ' -z 16-24 -w leaflet -o tms'
                    print(batcmd)
                    os.system(batcmd)
