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

import time
import cv2
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans


from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_blobs

#import rasterio
import gdal
from osgeo import gdalnumeric

#Liederik groene pixels sample:
#band1, band2, band3 = [90,127,74]
#band1, band2, band3 = [106,144,86]
#band1, band2, band3 = [244,252,221]
#band1, band2, band3 = [65,82,54]

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
#read image
src = gdal.Open(r"E:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Rijweg stalling 2\20190419\Orthomosaic/c01_verdonk-Rijweg stalling 2-20190419_clipped.tif")

x_block_size = 512
y_block_size = 512

#list to create subsest of blocks
it = list(range(0,5000, 10))

# Function to read the raster as arrays for the chosen block size.
def read_raster(x_block_size, y_block_size):
    tic = time.time()
    i = 0
    raster = r"E:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Rijweg stalling 2\20190419\Orthomosaic/c01_verdonk-Rijweg stalling 2-20190419_clipped.tif"
    #srcArray = gdalnumeric.LoadFile(raster)
    ds = gdal.Open(raster)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    template = np.zeros([ysize, xsize], np.uint8)
    #define kernel for morhpological closing operation
    kernel = np.ones((7,7), dtype='uint8')
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            blocks += 1
            #if statement for subset
            if blocks in it:
                if x + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - x
                b = np.array(src.GetRasterBand(1).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
                g = np.array(src.GetRasterBand(2).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
                r = np.array(src.GetRasterBand(3).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
                #array = ds.ReadAsArray(x, y, cols, rows)
                #array = array[0:3,:,:]
                if img.mean() > 0:
                    #array = array.reshape(array.shape[1], array.shape[2], array.shape[0])
                    #perform filtering on image to make plants and backgrond more uniform
                    img2 = cv2.medianBlur(img, ksize = 5)
                    #convert to CieLAB colorspace
                    img_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
                    a = np.array(img_lab[:,:,1])
                    b2 = np.array(img_lab[:,:,2])
                    #create input data array
                    a_flat = a.flatten()
                    b2_flat = b2.flatten()
                    Classificatie_Lab = np.column_stack((a_flat, b2_flat))
                    #perform kmeans clustering
                    #kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = 3)
                    kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = 3)
                    kmeans.fit(Classificatie_Lab)
                    y_kmeans = kmeans.predict(Classificatie_Lab)
                    #Get plants
                    y_kmeans[y_kmeans == 0] = 0
                    y_kmeans[y_kmeans == 1] = 0
                    y_kmeans[y_kmeans == 2] = 1
                    #convert output back to binary image                
                    kmeans_img = y_kmeans                
                    kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
                    binary_img = kmeans_img * 255
                    #close detected shapes
                    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
                    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
                    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
                    #write blocks on original sized image
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
                            #x,y,w,h = cv2.boundingRect(cnt)
                            output = img[bbox[1]-5: bbox[1]+bbox[3]+5, bbox[0]-5:bbox[0]+bbox[2]+5]
                            cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Training_data/image_'+str(i)+'.jpg', output)
                            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            cv2.drawMarker(img, (cx,cy), (0,0,255), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 1)
                            cv2.drawContours(output, cnt,-1, (255, 255, 255),-1)
                    #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(i)+'.jpg',img)     
                   # nr_of_img = create_training_data(img, closing, i)
                    template[y:y+rows, x:x+cols] = closing
                
    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijwegstalling2_blocks_test.jpg',template)     
    toc = time.time()
    print("processing took "+ str(toc - tic)+" seconds")

    #read bands
    b = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
    g = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
    r = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))
    
    #create img
    img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
    img[:,:,0] = b
    img[:,:,1] = g
    img[:,:,2] = r

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

    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\rijweg_plants_subset_test.jpg',img)

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
