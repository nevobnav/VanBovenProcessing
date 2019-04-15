# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:58:38 2019

@author: VanBoven
"""

# Imports
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import *
from geoalchemy2.shape import to_shape
import pandas as pd
import geopandas as gpd
import json
import os
import fiona
import datetime
os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')
from vanbovendatabase.postgres_lib import *

#specify customer name here:
customer_name = 'c07_hollandbean'

#database configuration
config_file_path = r'C:\Users\VanBoven\MijnVanBoven\config.json'

#output path and name
path = r'E:\VanBovenDrive\VanBoven MT\700 Data and Analysis\750 Percelen'

with open(config_file_path) as config_file:
    config = json.load(config_file)
DB_NAME = config['DB_NAME']
DB_USER = config['DB_USER']
DB_PASSWORD = config['DB_PASSWORD']
DB_IP = config['DB_IP']

def customer_plots2kml(customer_name, config_file_path, port):
    fiona.supported_drivers['KML'] = 'rw'
    cust_path = os.path.join(path, customer_name)
    if not os.path.exists(cust_path):
        os.makedirs(cust_path)
    con,meta = connect(DB_USER, DB_PASSWORD, DB_NAME, host=DB_IP)
    plot_ids = get_customer_plots(customer_name, meta, con)
    plot_names = get_customer_plots_name(customer_name, meta, con)
    for i, plot_id in enumerate(plot_ids):
        name = plot_names[i]
        name = name.replace(' ','_')
        geometry = to_shape(get_plot_shape(plot_id, meta, con))
        gdf = gpd.GeoDataFrame({'name': name, 'geometry': geometry}, index = [0])
        gdf.to_file(os.path.join(cust_path,str(name)+'.kml'), driver='KML')

#run to create kml files of plots
customer_plots2kml(customer_name, config_file_path, port)


