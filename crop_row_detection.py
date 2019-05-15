# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:04:11 2019

@author: VanBoven
"""

import os
os.chdir(r'C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing')

import pandas as pd

import time
import cv2
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
import math

import Random_Forest_Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_blobs

from scipy.spatial.distance import squareform, pdist


#import rasterio
import gdal
from osgeo import gdalnumeric

from sklearn import preprocessing
import numpy.ma as ma

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]]);
            image[y:y + windowSize[1], x:x + windowSize[0]] = window

def ExG(b,g,r):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    ExG_index = np.asarray((2.0*g - b - r), dtype=np.float32)
    ExG_index[ExG_index < 0] = 0
    ExG_index = np.asarray(scaler.fit_transform(ExG_index),dtype=np.uint8)    
    return ExG_index

from math import acos
from math import sqrt
from math import pi

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner
    
    
    
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

#read image
src = gdal.Open(r"E:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\NZ66_2\20190507\Orthomosaic/c08_biobrass-NZ66_2-20190507_clipped.tif")

b = src.GetRasterBand(1).ReadAsArray().astype(np.uint8)
g = src.GetRasterBand(2).ReadAsArray().astype(np.uint8)
r = src.GetRasterBand(3).ReadAsArray().astype(np.uint8)

img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
img[:,:,0] = b
img[:,:,1] = g
img[:,:,2] = r

#calc excess_green index
ExG_index = ExG(b,g,r)
#apply otsus thresholding to get most prominent green features
thresh, hough_test = cv2.threshold(ExG_index, 0, 255, cv2.THRESH_OTSU)
#apply morphological closing to make plants solid objects
kernel = np.ones((9,9), dtype='uint8')
closing = cv2.morphologyEx(hough_test, cv2.MORPH_CLOSE, kernel)
#remove features that are too large to be a plant
hough_ready = np.zeros([ExG_index.shape[0],ExG_index.shape[1]],dtype = np.uint8)
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
df = pd.DataFrame({'contours': contours})
df['area'] = df.contours.apply(lambda x:cv2.contourArea(x)) 
df = df[(df['area'] > 36) & (df['area'] < 750)]
df['moment'] = df.contours.apply(lambda x:cv2.moments(x))
df['coords'] = df.moment.apply(lambda x:[int(x['m10']/x['m00']),int(x['m01']/x['m00'])])
df['cx'] = df.moment.apply(lambda x:int(x['m10']/x['m00']))
df['cy'] = df.moment.apply(lambda x:int(x['m01']/x['m00']))
df.coords.apply(lambda x:cv2.drawMarker(hough_ready, (x[0],x[1]), (255), cv2.MARKER_STAR, markerSize = 5, thickness = 2))

#angle_array = pd.DataFrame(pdist(df.iloc[:,-1:-2]))
#squareform(angle_clockwise())
#lines = cv2.HoughLinesP(hough_ready, 1, np.pi/180, 500, minLineLength=10, maxLineGap=10000)
#for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv2.line(hough_ready, (x1, y1), (x2, y2), (255, 255, 255), 3)
# Show result

#find angle of max line
lines = cv2.HoughLines(hough_ready, 1, np.pi / 180, 500, None, 0, 0)

index2 = np.argmax(lines[:,:,0])
angle = lines[:,:,1][index2]

stepSize = 5000
windowSize = (501, 501)

x_list = []
y_list = []
window_list = []

tic = time.time()
for (x, y, window) in sliding_window(hough_ready, stepSize, windowSize):
    window_lines = cv2.HoughLines(window, 35, np.pi / 180,5, None, 0, 0)
    if window_lines is not None:
        window_lines = window_lines[window_lines[:,:,1] > angle - 0.02]
        window_lines = window_lines[window_lines[:,1] < angle + 0.02]
        for i in range(0, len(window_lines)):        
            #rho = lines[i][0]
            #theta = lines[i][1]
            rho = window_lines[i][0]
            theta = window_lines[i][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = (a * rho)
            y0 = (b * rho)
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            x0 = x0 + x
            y0 = y0 + y
            pt1 = (int(x0 + 5001*(-b)), int(y0 + 5001*(a)))
            pt2 = (int(x0 - 5001*(-b)), int(y0 - 5001*(a)))
            cv2.line(hough_ready, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
toc = time.time()
print("processing took "+ str(toc - tic)+" seconds")





df = pd.DataFrame({'x':x_list, 'y':y_list, 'window':window_list})    
df['lines'] = df.window.apply(lambda x:cv2.HoughLines(x, 25, np.pi / 180, 1, None, 0, 0))
for i in range(0, len(df['lines'])):
    if df.lines[i] is not None:
        for j in df.lines[i]:
            rho = j[:,0]
            theta = j[:,1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 5001*(-b)), int(y0 + 5001*(a)))
            pt2 = (int(x0 - 5001*(-b)), int(y0 - 5001*(a)))
            cv2.line(df.window[i], pt1, pt2, (255,255,255), 3, cv2.LINE_AA)

df.apply(lambda x: hough_ready[x.y:x.y + x.window.shape[1], x.x:x.x + x.window.shape[0]windowSize] = x.window)


toc = time.time()
print("processing took "+ str(toc - tic)+" seconds")


eucl_dist = pd.DataFrame(squareform(pdist(df.iloc[0:5000, 5:6])), columns = df.iloc[0:5000].index.unique(), index = df.iloc[0:5000].index.unique())
filter_eucl_dist = eucl_dist[eucl_dist< 500]



for col in test.columns:
    test.col.apply(lambda x)
    print(col)
    
test = np.asarray(squareform(pdist(df.iloc[1000:5000,5:6])), dtype = np.int32)


eucl_dist = np.asarray(squareform(pdist(df.iloc[1000:5000,5:6])),dtype = np.int32)
 
test = np.zeros([eucl_dist.shape[0],eucl_dist.shape[1]])
for i in range((3)):
    while i < (3) - 1:
        for j in range((eucl_dist.shape[0]) - i):
            pt1 = df.coords.iloc[i]
            pt2 = df.coords.iloc[j+i]
            angle_clock = angle_clockwise(pt1, pt2)
            test[i,j+i] = angle_clock



for i in range((eucl_dist.shape[0])):
    while i < (eucl_dist.shape[0]) - 1:
        for j in range((eucl_dist.shape[0]) - i):
            pt1 = df.coords.iloc[i]
            pt2 = df.coords.iloc[j+i]
            angle_clock = angle_clockwise(pt1, pt2)
            test[i,j+i] = angle_clock


line = df['lines']


df = df[df['lines'][:,1] > angle - 0.002]
df = df[df['lines'][:,1] < angle + 0.002]
df['a'] = df.lines.apply(lambda x:math.cos(x[:,1]))
df['b'] = df.lines.apply(lambda x:math.sin(x[:,1]))
df['x0'] = df.apply(lambda x:x['a'] * x['lines'][:,0])




    window_lines = cv2.HoughLines(window, 10, np.pi / 180, 1, None, 0, 0)
    if window_lines is not None:
        window_lines = window_lines[window_lines[:,:,1] > angle - 0.002]
        window_lines = window_lines[window_lines[:,1] < angle + 0.002]
        for i in range(0, len(window_lines)):        
            #rho = lines[i][0]
            #theta = lines[i][1]
            rho = lines[i][:,0]
            theta = lines[i][:,1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = x+(a * rho)
            y0 = y+(b * rho)
            pt1 = (int(x0 + 5001*(-b)), int(y0 + 5001*(a)))
            pt2 = (int(x0 - 5001*(-b)), int(y0 - 5001*(a)))
            cv2.line(window, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
            





  thetas = theta
  width, height = lines_img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(np.int32)   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2).astype(np.int32)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(lines_img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1






lines = lines[lines[:,:,1] > angle - 0.001]
lines = lines[lines[:,1] < angle + 0.001]

if lines is not None:
    for i in range(0, len(lines)):        
        rho = lines[i][0]
        theta = lines[i][1]
        #rho = lines[i][:,0]
        #theta = lines[i][:,1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 100000*(-b)), int(y0 + 100000*(a)))
        pt2 = (int(x0 - 100000*(-b)), int(y0 - 100000*(a)))
        cv2.line(img, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)




cv2.imwrite(r"E:\400 Data analysis\430 Filtering/ExG_index.jpg" , ExG_index) 
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/ExG_index_otsu.jpg" , hough_test) 
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/ExG_index_otsu_closed_NZ66_2.jpg" , closing)
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/ExG_index_otsu_closed_hough_ready_lines.jpg" , hough_ready)
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/ExG_index_otsu_closed_hough_lines_img.jpg" , img)





mask = img[img[:,:,1] == 0] = False
 

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
lab_mask = ma.masked_values(lab, 0)


ndvi_lab = np.asarray(scaler.fit_transform((lab_mask[:,:,1]-lab_mask[:,:,2])/(lab_mask[:,:,1]+lab_mask[:,:,2])) * 255, np.uint8)

thresh, hough_test = cv2.threshold(ndvi_lab, 0, 255, cv2.THRESH_OTSU)

lines_img = np.copy(edges)
lines_img = np.copy(hough_test)

lines = cv2.HoughLines(lines_img, 1, np.pi / 180, 500, None, 0, 0)

#rho = lines[:,:,0]
#theta = lines[:,:,1]

#hist = np.histogram(theta, bins = 360)

#index = np.argmax(hist[0])
#angle = hist[1][index]

index2 = np.argmax(lines[:,:,0])
angle = lines[:,:,1][index2]

if lines is not None:
    for i in range(0, len(lines)):        
        #rho = lines[i][0]
        #theta = lines[i][1]
        rho = lines[i][:,0]
        theta = lines[i][:,1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 100000*(-b)), int(y0 + 100000*(a)))
        pt2 = (int(x0 - 100000*(-b)), int(y0 - 100000*(a)))
        cv2.line(img, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)

spacing = 200
step = spacing
while spacing < 25000:
    pt1 = (int((x0+spacing) + 100000*(-b)), int((y0+spacing) + 100000*(a)))
    pt2 = (int((x0+spacing) - 100000*(-b)), int((y0+spacing) - 100000*(a)))
    pt3 = (int((x0-spacing) + 100000*(-b)), int((y0-spacing) + 100000*(a)))
    pt4 = (int((x0-spacing) - 100000*(-b)), int((y0-spacing) - 100000*(a)))
    cv2.line(img, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
    cv2.line(img, pt3, pt4, (255,255,255), 3, cv2.LINE_AA)
    spacing += step
    
cv2.imwrite(r'E:\400 Data analysis\420 Crop rows/anisodiff_crop_rows3.jpg', img)



lines = cv2.HoughLines(lines_img, 5, np.pi / 180, 1, None, 0, 0)


lines = lines[lines[:,:,1] > angle - 0.005]
lines = lines[lines[:,1] < angle + 0.005]


    
            
            
cv2.imwrite(r'E:\400 Data analysis\420 Crop rows/anisodiff_crop_rows2.jpg', ExG_index)


def hough_line(img, theta):
  # Rho and Theta ranges
  #thetas = np.deg2rad(np.arange(-90.0, 90.0))
  thetas = theta
  width, height = lines_img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(np.int32)   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2).astype(np.int32)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(lines_img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos

accumulator, thetas, rhos = hough_line(lines_img)


# Create binary image and call hough_line
image = np.zeros((50,50))
image[10:40, 10:40] = np.eye(30)
accumulator, thetas, rhos = hough_line(image)

# Easiest peak finding based on max votes
idx = np.argmax(accumulator)
rho = rhos[idx / accumulator.shape[1]]
theta = thetas[idx % accumulator.shape[1]]
print "rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta))




x_block_size = 512
y_block_size = 512

#list to create subsest of blocks
it = list(range(0,5000, 50))
#skip = True if you do not want to process each block but you want to process the entire image
skip = False

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
            b = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
            g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
            r = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows)).astype(np.uint(8))
            img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
            img[:,:,0] = b
            img[:,:,1] = g
            img[:,:,2] = r
            #cv2.imwrite(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2\blocks\rijwegstalling2_blocks_'+str(x)+'-'+str(y)+'.jpg',img)     
            #array = ds.ReadAsArray(x, y, cols, rows)
            #array = array[0:3,:,:]
            if img.mean() > 0:

                
                
                
                
                
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
grayscale = grayscale[10000:20000,10000:20000]

edges = cv2.Canny(grayscale, 500, 850)



cv2.imwrite(r"E:\400 Data analysis\430 Filtering/grayscale.jpg" , grayscale)
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/anisodiff2.jpg" , imgout)
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/anisodiff2_edges2.jpg" , edges)
cv2.imwrite(r"E:\400 Data analysis\430 Filtering/ExG_index.jpg" , ExG_index)                
                
def anisodiff(img,niter=100,kappa=35,gamma=0.2,step=(1.,1.),option=1,ploton=False):
        """
        Anisotropic diffusion.
  
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
  
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
  
        Returns:
                imgout   - diffused image.
  
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
  
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
  
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
  
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
  
        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
  
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
  
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
  
        June 2000  original version.      
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
  
        # ...you could always diffuse each color channel independently if you
        # really want
        if img.ndim == 3:
                warnings.warn("Only grayscale images allowed, converting to 2D matrix")
                img = img.mean(2)
  
        # initialize output array
        #img = img.astype('float32')
        imgout = img.copy()
  
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
  
        # create the plot figure, if requested
        if ploton:
                import pylab as pl
                from time import sleep
  
                fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
                ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
  
                ax1.imshow(img,interpolation='nearest')
                ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
                ax1.set_title("Original image")
                ax2.set_title("Iteration 0")
  
                fig.canvas.draw()
  
        for ii in range(niter):
  
                # calculate the diffs
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
  
                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
  
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
  
                # subtract a copy that has been shifted 'North/West' by one
                # pixel. don't as questions. just do it. trust me.
                NS[:] = S
                EW[:] = E
                NS[1:,:] -= S[:-1,:]
                EW[:,1:] -= E[:,:-1]
  
                # update the image
                imgout += gamma*(NS+EW)
  
                if ploton:
                        iterstring = "Iteration %i" %(ii+1)
                        ih.set_data(imgout)
                        ax2.set_title(iterstring)
                        fig.canvas.draw()
                        # sleep(0.01)
  
        return imgout        