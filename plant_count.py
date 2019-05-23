# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:07:25 2019

@author: VanBoven
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:46:57 2019

@author: ericv
"""
from tensorflow.keras import models

import os
os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')

import pandas as pd

import time
import cv2
import sklearn
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans

import Random_Forest_Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_blobs

from PIL import Image

#import rasterio
import gdal
from osgeo import gdalnumeric
from osgeo import ogr, osr
from fiona.crs import from_epsg
import fiona
import geopandas as gpd
import rasterio 
from rasterio.features import shapes
from shapely.geometry import mapping, Polygon, shape, Point

from functools import partial
from shapely.ops import transform
import pyproj

    
from skimage.util.shape import view_as_blocks

#Liederik groene pixels sample:
#band1, band2, band3 = [90,127,74]
#band1, band2, band3 = [106,144,86]
#band1, band2, band3 = [244,252,221]
#band1, band2, band3 = [65,82,54]

shp_dir = r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 1'
shp_name = 'test_points.shp'

#resize input images
def resize(x):
    new_shape = (50,50,3)
    x_resized = np.array(Image.fromarray(x).resize((50,50)))
    #X_train_new = scipy.misc.imresize(x, new_shape)
    x_rescaled = x_resized/255
    return x_rescaled

def points_in_array(x):
    for point in x:
        point_list = point[0]
        return cx, cy


def transform_geometry(geometry):
    project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:4326'), # source coordinate system
    pyproj.Proj(init='epsg:28992')) # destination coordinate system
    geometry = transform(project, geometry)  # apply projection
    return geometry

def write_plants2shp(img_path, df, shp_dir, shp_name):
    #get transform
    src = rasterio.open(img_path)
    #convert centroids to coords and contours to shape in lat, lon
    df['coords'] = df.centroid.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = x[0], cols = x[1], offset='ul'))
    df['geom'] = df.contours.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = list(x[:,0,1]), cols = list(x[:,0,0]), offset='ul'))
    #convert df to gdf
    #for polygon, first reformat into lists of coordinate pairs
    shape_list = []
    for geom in df.geom:
        x_list = geom[0]
        y_list = geom[1]
        coords_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            coords_list.append([x, y])
        shape_list.append(coords_list)
    df['geom2'] = shape_list

    #create points
    gdf_point = gpd.GeoDataFrame(df, geometry = [Point(x, y) for x, y in df.coords], crs = {'init': 'epsg:4326'})
    gdf_point = gdf_point.drop(['contours', 'moment', 'cx', 'cy', 'bbox', 'output', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)    
    #create polygons
    gdf_poly = gpd.GeoDataFrame(df, geometry = [Polygon(shape) for shape in df.geom2], crs = {'init': 'epsg:4326'}) 
    gdf_poly = gdf_poly.drop(['contours', 'moment', 'cx', 'cy', 'bbox', 'output', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)
    
    gdf_point.to_file(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2/AZ74.shp')
    gdf_poly.to_file(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2/AZ74_shape.shp')    

    return

    #get transform
    src = rasterio.open(img_path)
    #create mask
    mask = ma.masked_values(plant_contours, 0)    

    #vectorize
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(plant_contours, mask=mask, connectivity=8, transform=src.transform)))
    geoms = list(results)     
    #vectorize raster
    
    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'LAI': 'float:10.5',
                       'Height': 'int',
                       'Diameter':'float:10.5'},
    }
    schema2 = {
        'geometry': 'Point',
        'properties': {#'LAI': 'float:10.5',
                       #'Height': 'int',
                       'id':'int'},
    }
    
    #create output filename
    outfile = os.path.join(shp_dir, shp_name)
    #outfile = outfile[:-4] + ".shp"
    
    # Write a new Shapefile
    with fiona.open(str(outfile), 'w', 'ESRI Shapefile', schema2, crs = from_epsg(4326)) as c:
        ## If there are multiple geometries, put the "for" loop here
        for i in range(len(geoms)):
            geom = shape(geoms[i]['geometry'])
            coords = geom.centroid
            #geometry = transform_geometry(geom)
            #bounds = geometry.bounds
            c.write({
                'geometry': mapping(coords),
                'properties': {'id':i},
            })    
    
    with fiona.open(str(outfile), 'w', 'ESRI Shapefile', schema, crs = from_epsg(4326)) as c:
        ## If there are multiple geometries, put the "for" loop here
        for i in range(len(geoms)):
            geom = shape(geoms[i]['geometry'])
            geometry = transform_geometry(geom)
            bounds = geometry.bounds
            c.write({
                'geometry': mapping(geom),
                'properties': {'LAI': geometry.area, 
                               'Height': 0,
                               'Diameter': float(np.max([abs(bounds[2]-bounds[0]),abs(bounds[3]-bounds[1])]))},
            })    
      
    return

"""
#samples uit Liederik mbv qgis
sample_set = np.array([[[90,127,74],[106,144,86]],[[244,252,221],[65,82,54]]]).astype(np.uint8)
sample_lab = cv2.cvtColor(sample_set, cv2.COLOR_BGR2LAB)

#maak initiële waarden voor clustering
shadow_lab = cv2.cvtColor(np.array([[[14,18,17]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
#shadow_lab = cv2.cvtColor(np.array([[[10,10,10]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
light_lab = cv2.cvtColor(np.array([[[206,198,190]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
#light_lab = cv2.cvtColor(np.array([[[220,220,220]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
green_lab = cv2.cvtColor(np.array([[[126.25,151.25,108.75]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)

#convert to np array
shadow_init = np.array(shadow_lab[0,0,1:3])
light_init = np.array(light_lab[0,0,1:3])
green_init = np.array(green_lab[0,0,1:3])

kmeans_init = np.array([shadow_init, light_init, green_init])
"""
#values for AB_Vakwerk image
#maak initiële waarden voor clustering
shadow_lab = cv2.cvtColor(np.array([[[0,0,0]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
#shadow_lab = cv2.cvtColor(np.array([[[10,10,10]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
light_lab = cv2.cvtColor(np.array([[[255,255,255]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
#light_lab = cv2.cvtColor(np.array([[[220,220,220]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
green_lab = cv2.cvtColor(np.array([[[100,255,150]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)

#convert to np array
shadow_init = np.array(shadow_lab[0,0,1:3])
light_init = np.array(light_lab[0,0,1:3])
green_init = np.array(green_lab[0,0,1:3])

kmeans_init = np.array([shadow_init, light_init, green_init])

#ToDo:
"""
1. werk met raster blocks van 256*256 (meest efficient)
2. zorg dat NoData values buiten beschouwen worden gelaten
3. zie deze thread voor code: https://gis.stackexchange.com/questions/172666/optimizing-python-gdal-readasarray
4. en zie deze: https://github.com/OSGeo/gdal/issues/869

5. werk met een stop criterium voor kmeans
6. zorg dat gedurende het verwerken van de foto de init waarden worden geupdate om het aantal iteraties te laten af te nemen 
7. zoek uit wat voor filtering het beste werkt
8. kijk welk aantal init waarden (clusters) het beste werkt. 3 lijkt wat te weinig nu
"""

#train_data_path = r'E:\400 Data analysis\410 Plant count\Training_data'

#get trained RFC model
#model, scaler = Random_Forest_Classifier.get_trained_model(train_data_path)

#read image
#src = gdal.Open(r"E:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\NZ66_2\20190507\Orthomosaic/c08_biobrass-NZ66_2-20190507_clipped.tif")

#output path and name (voor nu nog gewoon zelf opgeven tot het operationeel is)
output_path = r'E:\400 Data analysis\410 Plant count\conv_net_test_classificatie.jpg'

#load model
model = models.load_model(r'C:\Users\VanBoven/cats_and_dogs_small_1.h5')

#
min_plant_size = 16
max_plant_size = 525

#set block_size
#x_block_size = 4096
#y_block_size = 4096

x_block_size = 512  
y_block_size = 512

#img_path
img_path = r'E:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190513\1357\Orthomosaic/c08_biobrass-AZ74-201905131357.tif'

#list to create subsest of blocks
it = list(range(0,400000, 3))
#skip = True if you do not want to process each block but you want to process the entire image
process_full_image = True
# Function to read the raster as arrays for the chosen block size.
def count_plants_in_image(x_block_size, y_block_size, model, process_full_image, it, img_path):    
    tic = time.time()
    i = 0
    #srcArray = gdalnumeric.LoadFile(raster)
    ds = gdal.Open(img_path)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
       
    template = np.zeros([ysize, xsize], np.uint8)
    plant_contours = np.zeros([ysize, xsize], np.uint8)
    #define kernel for morhpological closing operation
    kernel = np.ones((7,7), dtype='uint8')
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y > 0:
            y = y #- 50
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            if x > 0:
                x = x #- 50
            blocks += 1
            #if statement for subset
            if blocks in it:
                if x + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - x
                b = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
                g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
                r = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                #create empty array to store plant outline
                plant_contours_temp = np.zeros([img.shape[0], img.shape[1]], dtype = np.uint8)
                #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
                #array = ds.ReadAsArray(x, y, cols, rows)
                #array = array[0:3,:,:]
                if img.mean() > 0:
                    #array = array.reshape(array.shape[1], array.shape[2], array.shape[0])
                    #perform filtering on image to make plants and backgrond more uniform
                    
                    img = ma.masked_equal(img, 0)	#Mask an array where equal to a given value.
                    #mask = ma.masked_equal(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0)
                    
                    img2 = img                             
                    #img2 = cv2.medianBlur(img, ksize = 5)
                    
                    #convert to CieLAB colorspace
                    img_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
                    a = np.array(img_lab[:,:,1])
                    b2 = np.array(img_lab[:,:,2])
                    
   #                 a = ma.array(a)
  #                  a[mask] = ma.masked
                    
 #                   b2 = ma.array(b2)
#                    b2[mask] = ma.masked
                  
                    #create input data array
                    #test_a = ma.array(a)
                    #test_a[mask] = ma.masked
                    a_flat = a.flatten()
                    b2_flat = b2.flatten()
                    Classificatie_Lab = np.ma.column_stack((a_flat, b2_flat))
                    #perform kmeans clustering
                    kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = 3, verbose = False)
                    kmeans.fit(Classificatie_Lab)
                    y_kmeans = kmeans.predict(Classificatie_Lab)
                    #Get plants
                    y_kmeans[y_kmeans == 0] = 0
                    y_kmeans[y_kmeans == 1] = 0
                    y_kmeans[y_kmeans == 2] = 1
                    #convert output back to binary image                
                    kmeans_img = y_kmeans                
                    kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
                    binary_img = kmeans_img * 125
                    #close detected shapes
                    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
                    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
                    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
                    #write blocks on original sized image
                    
                    if process_full_image == False:
                        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        #template = np.zeros(img2.shape).astype(np.uint8)
                        for cnt in contours:
                            i+=1
                            ar = cv2.contourArea(cnt)
                            if (ar > min_plant_size) & (ar < max_plant_size):    
                                M = cv2.moments(cnt)
                                try:
                                    cx = int(M['m10']/M['m00'])
                                    cy = int(M['m01']/M['m00'])
                                except:
                                    print('0')
                                bbox = cv2.boundingRect(cnt)
                                #x,y,w,h = cv2.boundingRect(cnt)
                                output = img[bbox[1]-5: bbox[1]+bbox[3]+5, bbox[0]-5:bbox[0]+bbox[2]+5]
                                if output.shape[0] * output.shape[1] > 0:
                                    #prediction = model.predict(output)
                                    #output_features = Random_Forest_Classifier.get_image_features(output, scaler)                                
                                    #prediction = str(model.predict(output_features)[0])
                                    #print(prediction)
                                    #if predict_proba[0][1] >= 0.3 and predict_proba[0][3] < 0.8:
                                        #prediction = 'Broccoli'
                                    #if prediction != 'Cover':#== 'Broccoli' or prediction == 'Grass':                             
                                        #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Training_data/image_'+str(i)+'.jpg', output)
                                        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                                        #cv2.drawMarker(img, (cx,cy), (255,0,0), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 1)
                                        #cv2.drawContours(img, cnt,-1, (255, 255, 255),-1)
                                        #cv2.drawMarker(plant_contours_temp, (cx,cy), (255,0,0), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 1)
                                    cv2.drawContours(plant_contours_temp, [cnt],-1, (255, 255, 255),-1)
                                    #else:
                                     #   continue
                                        #cv2.drawMarker(img, (cx,cy), (0,0,255), cv2.MARKER_STAR, markerSize = 5, thickness = 1)
                        #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 1\rijwegstalling1_blocks_'+str(i)+'.jpg',img)     
                    # nr_of_img = create_training_data(img, closing, i)
                    #plant_contours[y:y+rows, x:x+cols] = plant_contours[y:y+rows, x:x+cols] + plant_contours_temp                    
                    #write_plants2shp(img_path, plant_contours, shp_dir, shp_name)                    
                    template[y:y+rows, x:x+cols] = template[y:y+rows, x:x+cols] + closing
                    print('processing of block ' + str(blocks) + ' finished')
    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c03_termote\10m_test_binnendijk_links'+str(i)+'.jpg',template)     
    plant_contours[plant_contours > 0] = 255
    toc = time.time()
    print("processing of blocks took "+ str(toc - tic)+" seconds")
    if process_full_image == True:

        print('Start with classification of objects')
        #initiate output img
        output = np.zeros([ysize,xsize,3], np.uint8)
        b = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
        output[:,:,0] = b
        b = None
        g = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
        output[:,:,1] = g
        g = None
        r = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))
        output[:,:,2] = r
        r = None
        
        result_img = np.zeros((template.shape[0],template.shape[1]),dtype=np.uint8)
        
        #Get contours of features
        contours, hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #create df with relevant data
        df = pd.DataFrame({'contours': contours})
        df['area'] = df.contours.apply(lambda x:cv2.contourArea(x)) 
        df = df[(df['area'] > 81) & (df['area'] < 500)]
        df['moment'] = df.contours.apply(lambda x:cv2.moments(x))
        df['centroid'] = df.moment.apply(lambda x:(int(x['m01']/x['m00']),int(x['m10']/x['m00'])))
        df['cx'] = df.moment.apply(lambda x:int(x['m10']/x['m00']))
        df['cy'] = df.moment.apply(lambda x:int(x['m01']/x['m00']))
        df['bbox'] = df.contours.apply(lambda x:cv2.boundingRect(x))
        #create input images for model
        df['output'] = df.bbox.apply(lambda x:output[x[1]-5: x[1]+x[3]+5, x[0]-5:x[0]+x[2]+5])
        df = df[df.output.apply(lambda x:x.shape[0]*x.shape[1]) > 0]
        #resize data to create input tensor for model
        df['input'] = df.output.apply(lambda x:resize(x))
        
        #remove img from memory
        output = None
        
        #resize images
        #df.input.apply(lambda x:x.resize(50,50,3, refcheck=False))       
        model_input = np.asarray(list(df.input.iloc[:]))
        
        #predict
        tic = time.time()
        prediction = model.predict(model_input)
        #get prediction result
        pred_final = prediction.argmax(axis=1)
        #add to df
        df['prediction'] = pred_final
        toc = time.time()
        print('classification of '+str(len(df))+' objects took '+str(toc - tic) + ' seconds')
        
        write_plants2shp(img_path, df, shp_dir, shp_name)
        

        #Create mask with predictions
        for i in range(len(df)):
            cv2.drawContours(result_img, [df.contours.iloc[i]],-1, (df.prediction.iloc[i]+1),-1)
        

        write_plants2shp(img_path, )
        
        
        for cnt in contours:
            tic2 = time.time()
            #get area of each contour
            ar = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            except:
                continue
                #print('0')
            if (ar > min_plant_size) & (ar < max_plant_size):   
                bbox = cv2.boundingRect(cnt)
                feature = output[bbox[1]-5: bbox[1]+bbox[3]+5, bbox[0]-5:bbox[0]+bbox[2]+5]
                if feature.shape[0] * feature.shape[1] > 0:                                                      
                    output_features = Random_Forest_Classifier.get_image_features(feature, scaler)                                
                    prediction = str(model.predict(output_features)[0])
                    #if predict_proba[0][1] >= 0.3 and predict_proba[0][3] < 0.8:
                        #prediction = 'Broccoli'
                    if prediction == 'Broccoli' or prediction == 'Grass':                             
                        #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Training_data/image_'+str(i)+'.jpg', output)
                        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.drawMarker(output, (cx,cy), (255,0,0), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 1)
                        cv2.drawContours(output, cnt,-1, (255, 255, 255),-1)
                    else:
                        continue    
    cv2.imwrite(output_path, output)
    
    
    
    
    
   


img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
img = ma.masked_equal(img, 255)	#Mask an array where equal to a given value.
mask = img.mask
mask = np.dstack((mask,mask,mask))

output[mask] = 0

mask = img.mask.reshape(img.shape[0], img.shape[1], 3)

output = np.ma.array(output, mask = mask)

def make_blocks_vectorized_v2(x,d):
    return view_as_blocks(x,(d,d, x.shape[2])).squeeze()

def get_windows(x, d):
    #create array with shape devidable by d
    z = np.zeros(((int(x.shape[0]/32)+1)*d,(int(x.shape[1]/32)+1)*d, 3), dtype = np.uint8)
    #set values of x to new array
    z[:x.shape[0],:x.shape[1],:] = x
    z = make_blocks_vectorized_v2(z,d)
    z = z.reshape((z.shape[0]*z.shape[1]),d,d,3)
    return z



#z = np.zeros((3, (int(x.shape[1]/32)+1)*d,(int(x.shape[2]/32)+1)*d), dtype = np.uint8)
#z[:,:x.shape[1],:x.shape[2]] = x
d = 32
z = get_windows(output,d)




j = list(range(0,z.shape[0]-25000, 25000))
k = list(range(25000, z.shape[0], 25000)) 
test2 = []
for i in range(1):#len(j)):
    chunk = z[j[i]:k[i],:,:,:]
    for l in range(chunk.shape[0]):
        temp = resize(chunk[l,:,:,:])
        if temp.mean() < 1:
            test2.append(temp)

model_input = np.asarray(test2)

test2 = pd.DataFrame({'img':test2})
test2.img.apply(lambda x:resize(x))  
model_input = np.asarray(list(test2.img.iloc[:]))

tic = time.time()
prediction = model.predict(reshaped[200000:700000,:,:,:])
toc= time.time()
print('classification of '+str(len(df))+' objects took '+str(toc - tic) + ' seconds')






x = cv2.imread(r'E:\400 Data analysis\410 Plant count\Training_data\Broccoli/AZ74_190513_21693.jpg',-1)
x = resize(x)
x = x[np.newaxis,:]
prediction = model.predict(x)

from tensorflow.keras import backend as K

# This is the "broccoli" entry in the prediction vector
broccoli_output = model.output[:, 1]
# This is the output feature map of
# the last convolutional layer in our model
last_conv_layer = model.get_layer(index=-6)
# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(broccoli_output, last_conv_layer.output)[0]
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)








































    
    toc = time.time()
                     

        
                                            
    toc = time.time()
    print("processing took "+ str(toc - tic)+" seconds")
    return template

    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\NZ_2_closed.jpg',template) 
    
    toc = time.time()
    print("processing took "+ str(toc - tic)+" seconds")

    #read bands
    b = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    g = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint8)
    r = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint8)
    
    #create img
    img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
    img[:,:,0] = b
    img[:,:,1] = g #<-- Lekker bezig G
    img[:,:,2] = r
    
    #temp voor development     
    #template = cv2.imread(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2/rijweg_closing.jpg')
    #template = template[:,:,0]
    #img = cv2.imread(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2/rijweg_plants_outline.jpg')
    
    contours, hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img2 = np.zeros([template.shape[0],template.shape[1]])
    """
    tic = time.time()
    i = 0
    #template = np.zeros(img2.shape).astype(np.uint8)
    df = pd.DataFrame({'contours': contours})
    df['area'] = df.contours.apply(lambda x:cv2.contourArea(x)) 
    df = df[(df['area'] > 81) & (df['area'] < 500)]
    df['moment'] = df.contours.apply(lambda x:cv2.moments(x))
    df['cx'] = df.moment.apply(lambda x:int(x['m10']/x['m00']))
    df['cy'] = df.moment.apply(lambda x:int(x['m01']/x['m00']))
    df['bbox'] = df.contours.apply(lambda x:cv2.boundingRect(x))
    df['output'] = df.bbox.apply(lambda x:img[x[1]-5: x[1]+x[3]+5, x[0]-5:x[0]+x[2]+5])
    df = df[df.output.apply(lambda x:x.shape[0]*x.shape[1]) > 0]

    output_features = df.output.apply(lambda x: Random_Forest_Classifier.get_image_features(x, scaler))
    prediction = output_features.apply(lambda x: str(model.predict(output_features)[0]))
    toc = time.time()
    print("processing took "+ str(toc - tic)+" seconds")

    
    
    output_features = Random_Forest_Classifier.get_image_features(output, scaler)                                
    prediction = str(model.predict(output_features)[0])
        
    """
    tick = 0
    tic = time.time()
    for cnt in contours:
        tic2 = time.time()
        #get area of each contour
        ar = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except:
            continue
            #print('0')
        if (ar > 9) & (ar < 1001):   
            bbox = cv2.boundingRect(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            output = img[bbox[1]-5: bbox[1]+bbox[3]+5, bbox[0]-5:bbox[0]+bbox[2]+5]
            if output.shape[0] * output.shape[1] > 0:
                output_features = Random_Forest_Classifier.get_image_features(output, scaler)                                
                prediction = str(model.predict(output_features)[0])
                #print(prediction)
                if prediction == 'Broccoli':                             
                    #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Training_data/image_'+str(i)+'.jpg', output)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    #i += 1
                    cv2.drawMarker(img2, (cx,cy), (255,255,255), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 2)
                    toc2 = time.time()
                    print('Classiying one Broccoli took '+str(toc2-tic2)+ ' seconds')
                    #cv2.drawContours(img, cnt,-1, (255, 255, 255),-1)
                #if prediction == 'Grass':
                    #cv2.drawMarker(img, (cx,cy), (0,0,255), cv2.MARKER_STAR, markerSize = 5, thickness = 2)                  
                #if prediction == 'Background':
                    #cv2.drawMarker(img, (cx,cy), (255,255,0), cv2.MARKER_STAR, markerSize = 5, thickness = 2)                  
                #else:
                    #cv2.drawMarker(img, (cx,cy), (255,0,0), cv2.MARKER_STAR, markerSize = 5, thickness = 2)
                print(tick)
                tick += 1

            #cv2.drawMarker(img, (cx,cy), (0,0,255), markerType = cv2.MARKER_STAR, markerSize = 9, thickness = 2)
            #cv2.drawContours(img, cnt,-1, (255, 255, 255),-1)

    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Broccoli_classification_NZ66_2.jpg',img2)
    toc = time.time()
    classification_time_NZ66_2 = toc - tic
    print('classification of broccoli took '+str(classification_time_NZ66_2)+ ' seconds')

    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Broccoli_train_NZ66_2.jpg',img[10000:13000,9000:12000])



def create_training_data(img, closing, i):
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #template = np.zeros(img2.shape).astype(np.uint8)
    for cnt in contours:
        i+=1
        ar = cv2.contourArea(cnt)
        if (ar > 9) & (ar < 4001):    
            M = cv2.moments(cnt)
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            except:
                print('0')
            bbox = cv2.boundingRect(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            output = img[bbox[1]-5: bbox[1]+bbox[3]+5, bbox[0]-5:bbox[0]+bbox[2]+5]
            cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Training_data/image_'+str(i)+'.jpg', output)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.drawMarker(img, (cx,cy), (0,0,255), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 1)
            cv2.drawContours(output, cnt,-1, (255, 255, 255),-1)
    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(i)+'.jpg',img)     
    return i
    
    





"""        
    gdalnumeric.SaveArray(srcArray, r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijwegstalling2_blocks_test.jpg', format="JPEG")            

            del array
            blocks += 1
    band = None
    ds = None
    print("{0} blocks size {1} x {2}:".format(blocks, x_block_size, y_block_size))

#read bands
b = np.array(src.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
g = np.array(src.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
r = np.array(src.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))

#create img
img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
img[:,:,0] = b
img[:,:,1] = g
img[:,:,2] = r

#convert image to 8bit integer
img = img.astype(np.uint8)

img2 = img[15000:30000,10000:25000,:]

#drop alpha band
#img = img[:,:, 0:3]

img_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
img_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
img_hls = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)

combined = (img_lab + img_hsv + img_hls)//3

cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_LAB.jpg',img_lab)
cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_HSV.jpg',img_hsv)
cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_HLS.jpg',img_hls)
cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_combined.jpg',combined)

L = np.array(img_lab[:,:,0])
a = np.array(img_lab[:,:,1])
b2 = np.array(img_lab[:,:,2])

L_flat = L.flatten()
a_flat = a.flatten()
b2_flat = b2.flatten()

Classificatie_Lab = np.column_stack((a_flat, b2_flat))

#generate random subset of 20% of the parcel
subset = Classificatie_Lab[np.random.randint(Classificatie_Lab.shape[0], size=int(len(Classificatie_Lab)/10)), :]


#normal k-means clustering
kmeans = KMeans(init = kmeans_init, n_jobs = 7, max_iter = 25, n_clusters = 3)
kmeans.fit(subset)
kmeans.fit(Classificatie_Lab)
y_kmeans = kmeans.predict(Classificatie_Lab)

#select clusters containing of plant pixels
y_kmeans[y_kmeans == 0] = 0
y_kmeans[y_kmeans == 1] = 0
y_kmeans[y_kmeans == 2] = 1

kmeans_img = y_kmeans

kmeans_img = kmeans_img.reshape(img2.shape[0:2]).astype(np.uint8)
binary_img = kmeans_img * 255

mask = np.zeros([1000,1000,3])
mask[:,:,0] = kmeans_img
mask[:,:,1] = kmeans_img
mask[:,:,2] = kmeans_img
mask = mask.astype(np.uint8)

test = np.ma.masked_array(img, mask)

kmeans_img2 = img2

kmeans_img2[:,:,0] = kmeans_img2[:,:,0] * kmeans_img
kmeans_img2[:,:,1] = kmeans_img2[:,:,1] * kmeans_img
kmeans_img2[:,:,2] = kmeans_img2[:,:,2] * kmeans_img
kmeans_img2[kmeans_img2[:,:,2] == 1] = 255
kmeans_img2[kmeans_img2[:,:,1] == 1] = 255
kmeans_img2[kmeans_img2[:,:,0] == 1] = 255

cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_kmeans_cluster_binary.jpg',kmeans_img2)

kernel = np.ones((9,9), dtype='uint8')

opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

edges = cv2.Canny(binary_img,50,100)

cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_closing.jpg',closing)
#find contours
contours, hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

i = 0
#template = np.zeros(img2.shape).astype(np.uint8)
for cnt in contours:
    #get area of each contour
    ar = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except:
        print('0')
    if (ar > 9) & (ar < 401):     
        i += 1
        cv2.drawMarker(img, (cx,cy), (0,0,255), markerType = cv2.MARKER_STAR, markerSize = 9, thickness = 2)
        cv2.drawContours(img, cnt,-1, (255, 255, 255),-1)
    
temp = template[:,:,0]

img3 = img2/2
img3[:,:,0] = img3[:,:,0] + temp
img3[:,:,1] = img3[:,:,1] + temp
img3[:,:,2] = img3[:,:,2] + temp

img3 = (img3/img3.max())*255
img3[img3 < 0] = 0

img3 = img3.astype(np.uint8)
"""
cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_template.jpg',template)
