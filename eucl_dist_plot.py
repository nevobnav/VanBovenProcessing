# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:03:29 2019

@author: VanBoven
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy
from sklearn.neighbors import NearestNeighbors


#max number of pixels in image is restricted, in order to open big orthos it has to be modified
Image.MAX_IMAGE_PIXELS = 3000000000      

template = np.array(Image.open(r'F:\700 Georeferencing\AZ74 georeferencing\plant_count/c08_biobrass-AZ74-201905131357_no_closing.jpg'), dtype = np.uint8)
zeros = np.zeros((template.shape[0], template.shape[1]), dtype = np.uint8)

#Get contours of features
contours, hierarchy = cv2.findContours(template, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
#create df with relevant data
df = pd.DataFrame({'contours': contours})
df['area'] = df.contours.apply(lambda x:cv2.contourArea(x)) 
df = df[(df['area'] > 16) & (df['area'] < 1500)]
df['moment'] = df.contours.apply(lambda x:cv2.moments(x))
df['centroid'] = df.moment.apply(lambda x:(int(x['m01']/x['m00']),int(x['m10']/x['m00'])))

X = np.array(list(df.centroid.iloc[:]))

knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2,radius=130.0).fit(X)

distances, indices = knn.kneighbors(X)

#distances in crop rows: 0.35 - 0.52
#distances 

#mask out points not matching criteria
mask = np.zeros((distances.shape[0]))
for i in range(distances.shape[0]):
    mask[i]=(distances[i,1] > 28 and distances[i, 2] < 62 and distances[i, 3] > 65 and distances[i,4] < 99)

it = list(range(0,1, 1))
for j in it:
    #draw points matching criteria
    points = X[np.where(mask == 1)]
    knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2,radius=100.0).fit(points)
    distances, indices = knn.kneighbors(X)
    mask = np.zeros((distances.shape[0]))
    for i in range(distances.shape[0]):
        mask[i]=(distances[i,1] > 28 and distances[i, 2] < 62 and distances[i, 3] > 65 and distances[i,4] < 92)
   
for point in points:
    cv2.drawMarker(zeros, (point[1],point[0]), (255), markerType = cv2.MARKER_STAR, markerSize = 5, thickness = 2)    
    
#points2 =np.asarray((points[:,1], points[:,0])).T
#plot_boundary = cv2.minAreaRect(points2)
#box = cv2.boxPoints(plot_boundary)
#box = np.int0(box)
#cv2.drawContours(zeros,[box],0,(255),2)
kernel = np.ones((1000,1000), dtype='uint8')
dilate = cv2.dilate(zeros, kernel, 10)

contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
#closing = cv2.morphologyEx(zeros, cv2.MORPH_CLOSE, kernel)
cv2.drawContours(zeros, [contours[0]],-1, (255),-1)

cv2.imwrite(r'F:\700 Georeferencing\AZ74 georeferencing\clipped_imagery/eucl_dist_zeros4.jpg',zeros)
cv2.imwrite(r'F:\700 Georeferencing\AZ74 georeferencing\clipped_imagery/eucl_dist_dilate4.jpg',dilate)



test = scipy.spatial.ConvexHull(points2)
cv2.line
