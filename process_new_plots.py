# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:58:38 2019

@author: VanBoven
"""

# Imports
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import *
import pandas as pd
import geopandas as gpd
import json
import os
import fiona
import datetime

#database configuration
config_file_path = r'C:\Users\VanBoven\MijnVanBoven\config.json'
port = 5432
table_name = 'portal_plot'

#shapefile path and name
path = r'E:\VanBovenDrive\VanBoven MT\700 Data and Analysis\750 Percelen'
filename = 'Percelen.shp'

#customer_id of plots, number only
customer_id = 8
plot_name_column = 'perceel'


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

def read_SHP(path, filename):
    gdf = gpd.read_file(os.path.join(path, filename))
    return gdf
    
def shp_2_kmls(gdf, plot_name_column):
    fiona.supported_drivers['KML'] = 'rw'
    for index, row in gdf.iterrows():
        name = row[str(plot_name_column)]
        plot = gpd.GeoDataFrame(row)
        plot = plot.T
        #plot['geometry'] = Geometry('POLYGON', srid= 4326)
        plot.to_file(os.path.join(path,str(name)+'.kml'), driver='KML')
    
def write_shp2DB(table_name, con):
    # Use 'dtype' to specify column's type
    # For the geom column, we will use GeoAlchemy's type 'Geometry'
    gdf.to_sql(table_name, con, if_exists='append', index=False, dtype={'geom': Geometry('POLYGON', srid= 4326)})    
    
def create_empty_gdf():
    empty_gdf = gpd.GeoDataFrame(columns = ['id', 'name', 'street', 'number', 'crop', 'startdate', 'customer_id', 'shape'])    
    return empty_gdf    


"""
See example below for how to usage the functions
"""      
# Creating SQLAlchemy's engine to use
con, meta = connect(config_file_path, port)
#some example code used for verdonk please note that null values are not accepted by the DB
#for proper usage check in db which id values to use for the new plots
gdf = read_SHP(path, filename)
empty_gdf = create_empty_gdf()
verdonk = gdf[(gdf['CustomerID'] == 'Verdegaal') | (gdf['CustomerID'] == 'verdegaal')]
empty_gdf['shape'] = verdonk['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))
empty_gdf['name'] = verdonk['perceel']
empty_gdf['customer_id'] = customer_id
empty_gdf['crop'] = 'Bollen'
empty_gdf['street'] = 'Onbekend'
empty_gdf['number'] = 1
#empty_gdf['street'] = ['Wijzenddijkje', 'Wijzenddijkje', 'Tropweere', 'Tropweere', 'Klaver', 'Knipping', 'Rijweg', 'Rijweg', 'Mammoet']
empty_gdf['startdate'] = datetime.date(2019, 3, 19)

empty_gdf.to_sql(table_name, con, if_exists='append', index=False, dtype={'shape': Geometry('POLYGON', srid= 4326)})    

shp_2_kmls(gdf, plot_name_column)
