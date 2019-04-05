# -*- coding: utf-8 -*-
"""
@author: Kaz
"""

import os
import sys
#import gdal2tiles
import pandas as pd
import subprocess
from gdal2tilesp import gdal2tilesp


## CONFIG SECTION ##
inbox = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\ready' #folder where all orthos are stored
files = [f for f in os.listdir(inbox) if f.endswith('.tif')]
print(files)

orthos = []
for f in files:
    this_customer,this_plot_name,this_date = files[0].split('-')
    dict = {"customer": this_customer, "plot_name":this_plot_name, "date":this_date}
    orthos.append(dict)



    batcmd = r'C:/Users/VanBoven/Documents/GitHub/VanBovenProcessing/gdal2tilesp/gdal2tilesp.py '\
    + '"' + str(input_file) + '"' + ' "' + str(output_file) + '"'+ ' -z 16-24 -w leaflet -o tms'
    os.system(batcmd)
                
