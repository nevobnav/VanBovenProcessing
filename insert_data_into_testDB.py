# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:25:02 2020

@author: VanBoven
"""

import geopandas as gpd
import os
import json
from geoalchemy2 import WKTElement
from geoalchemy2 import Geometry
from sqlalchemy.sql import select

import time


from sqlalchemy import event, create_engine

from vanbovendatabase.postgres_lib import *

#db connection info
config_file_path = r"C:\Users\VanBoven\MijnVanBoven\config_testDB.json"
port = 5432
with open(config_file_path) as config_file:
        config = json.load(config_file)
host=config.get("DB_IP")
db=config.get("DB_NAME")
user=config.get("DB_USER")
password=config.get("DB_PASSWORD")

con, meta = connect(user, password, db, host=host, port=port)

engine = con

@event.listens_for(engine, 'before_cursor_execute')
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    if executemany:
        cursor.fast_executemany = True
        cursor.commit()

#specific to plant table
gdf2 = gpd.read_file(r"C:\Users\VanBoven\Desktop\populate_testDB\plants.shp")
gdf2.columns = ['plant_detection_id', 'batch_id', 'geometry']
gdf2['location'] = gdf2['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))
gdf2.drop('geometry', 1, inplace=True)
gdf2 = gdf2.dropna()
gdf2['id'] = gdf2.index

gdf2.to_sql('database_plant', con, if_exists='append', index=False, index_label = 'id', dtype={'location': Geometry(geometry_type='POINT', srid= 4326)})


#specific to plant_detection
gdf = gpd.read_file(r"A:\c08_biobrass\Schol\20190726\1210\Plant_count\20190726_count_merged.shp")
gdf.columns = ['id', 'plant_detection_job_id', 'geometry']
gdf.head(10)
gdf.id = gdf.id.astype(int)
gdf['plant_detection_job_id'] = 1
gdf['plant_detection_job_id'] = gdf['plant_detection_job_id'].astype(int)

#https://gis.stackexchange.com/questions/239198/geopandas-dataframe-to-postgis-table-help  
gdf['location'] = gdf['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))


#drop the geometry column as it is now duplicative
gdf.drop('geometry', 1, inplace=True)

#get current plant detectionIDs
plant_detections = meta.tables['database_plant_detection']
query = select([plant_detections.c.id])
res = con.execute(query)
result = []
[result.append(x[0]-1) for x in res]

gdf.drop(result, 0, inplace=True)


tic = time.time()
gdf.to_sql('database_plant_detection', con, if_exists='append', index=False, index_label='id', dtype={'location': Geometry(geometry_type='POINT', srid= 4326)})
toc = time.time()
print('inserting data in DB took ' + str(toc-tic) + ' seconds')

gdf.to_file(r"C:\Users\VanBoven\Desktop\populate_testDB\plant_detection.shp")


gdf.columns = ['id', 'batchID', 'geometry']

cities_with_country = geopandas.sjoin(cities, countries, how="inner", op='intersects')

