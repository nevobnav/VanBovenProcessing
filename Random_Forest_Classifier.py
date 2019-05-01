# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:50:06 2019

@author: VanBoven
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import mahotas
import cv2
import numpy as np
import os
import time
import pandas as pd


train_data_path = r'E:\400 Data analysis\410 Plant count\Training_data'

#functions for feature extraction
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    try:
        haralick = mahotas.features.haralick(gray, ignore_zeros = False).mean(axis=0)
    except:
        print('lukte niet')
    return haralick
 
def fd_histogram(image, mask=None, bins = 255):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    hist.flatten()
    
def create_training_data(train_data_path):
    #create training data input for model
    tic = time.time()
    #load training data:
    X = []
    y_train = []
    for folder in os.listdir(train_data_path):
        if folder != 'Unclassified':
            input_folder = os.path.join(train_data_path, folder)
            for file in os.listdir(input_folder):
                if file.endswith('.jpg'):
                    image = mahotas.imread(os.path.join(input_folder, file))
                    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
                    #Normalize The feature vectors...
                    X.append(global_feature)
                    #check wether to use strings or integers
                    y_train.append(str(folder))
    
    X = np.asarray(X)     
    y_train = np.asarray(y_train)
    X = X[:,1:]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    toc = time.time()
    processing_time = toc - tic
    print('Importing training data took ' + str(processing_time) + ' seconds')
    return X, y_train, scaler
    
def train_RFC(X, y_train):    
    #train model:
    #random forest classification
    model = RandomForestClassifier(random_state=0)
    #fit model
    model.fit(X, y_train)
    return model

def get_trained_model(train_data_path):
    X, y_train, scaler = create_training_data(train_data_path)
    model = train_RFC(X, y_train)
    return model, scaler
    
def get_image_features(image, scaler):
    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
    #Normalize The feature vectors...
    #X.append(global_feature)    
    X = np.asarray(global_feature)     
    X = X[1:]
    X = X.reshape(1, -1)
    X = scaler.transform(X)
    return X

"""

mooi voorbeeld

#split data in training and validation
data = pd.DataFrame(X)   
clas = pd.DataFrame(y_train)
clas = clas.astype(str)

dataset = data
dataset['classificatie'] = clas

dataset['is_train'] = np.random.uniform(0, 1, len(dataset)) <= .75

df = dataset

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

model = RandomForestClassifier(random_state=0)

features = df.columns[:20]
y= train['classificatie']
#y = pd.factorize(train['classificatie'])[0]

model.fit(train[features], y)

preds = model.predict(test[features])
#preds = test.classificatie[model.predict(test[features])]

model.predict_proba(test[features])[0:25]

pd.crosstab(test['classificatie'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

list(zip(train[features], model.feature_importances_))

"""
