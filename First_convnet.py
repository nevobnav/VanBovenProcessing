from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow
import time, os, shutil
import random

import os, shutil
import matplotlib.pyplot as plt

import sys
from PIL import Image
sys.modules['Image'] = Image 

from PIL import Image
print(Image.__file__)

import Image
print(Image.__file__)

import os

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = r'E:\400 Data analysis\410 Plant count\Training_data'
# The directory where we will
# store our smaller dataset
base_dir = r'E:\400 Data analysis\410 Plant count\Training_data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

def create_train_and_validation_sets():
    #os.mkdir(base_dir)
    # Directories for our training,
    # validation and test splits
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'Broccoli')
    os.mkdir(train_cats_dir)
    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'Grass')
    os.mkdir(train_dogs_dir)
    # Directory with our training cover pictures
    train_cover_dir = os.path.join(train_dir, 'Cover')
    os.mkdir(train_cover_dir)
    # Directory with our training background pictures
    train_background_dir = os.path.join(train_dir, 'Background')
    os.mkdir(train_background_dir)
    
    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'Broccoli')
    os.mkdir(validation_cats_dir)
    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'Grass')
    os.mkdir(validation_dogs_dir)
    # Directory with our validation cat pictures
    test_cats_dir = os.path.join(test_dir, 'Broccoli')
    os.mkdir(test_cats_dir)
    # Directory with our validation dog pictures
    test_dogs_dir = os.path.join(test_dir, 'Grass')
    os.mkdir(test_dogs_dir)
    # Directory with our validation cat pictures
    validation_cover_dir = os.path.join(validation_dir, 'Cover')
    os.mkdir(validation_cover_dir)
    # Directory with our validation dog pictures
    validation_background_dir = os.path.join(validation_dir, 'Background')
    os.mkdir(validation_background_dir)
    # Directory with our validation cat pictures
    test_cover_dir = os.path.join(test_dir, 'Cover')
    os.mkdir(test_cover_dir)
    # Directory with our validation dog pictures
    test_background_dir = os.path.join(test_dir, 'Background')
    os.mkdir(test_background_dir)

    # Copy first 1000 cat images to train_cats_dir
    #fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

    Broccolinames = os.listdir(os.path.join(base_dir, 'Broccoli'))
    random.shuffle(Broccolinames)
    for fname in Broccolinames[:int(len(Broccolinames)/2)]:
        src = os.path.join(base_dir, 'Broccoli', fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 cat images to validation_cats_dir
    #fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in Broccolinames[int(len(Broccolinames)/2):int(len(Broccolinames)*0.75)]:
        src = os.path.join(base_dir, 'Broccoli', fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 cat images to test_cats_dir
    #fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in Broccolinames[int(len(Broccolinames)*0.75):]:
        src = os.path.join(base_dir, 'Broccoli', fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
        
    # Copy first 1000 dog images to train_dogs_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    Grassnames = os.listdir(os.path.join(base_dir, 'Grass'))
    random.shuffle(Grassnames)
    for fname in Grassnames[:int(len(Grassnames)/2)]:
        src = os.path.join(base_dir, 'Grass', fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to validation_dogs_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in Grassnames[int(len(Grassnames)/2):int(len(Grassnames)*0.75)]:
        src = os.path.join(base_dir, 'Grass', fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to test_dogs_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in Grassnames[int(len(Grassnames)*0.75):]:
        src = os.path.join(base_dir, 'Grass', fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
           
    # Copy first 1000 dog images to train_dogs_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    Covernames = os.listdir(os.path.join(base_dir, 'Cover'))
    random.shuffle(Covernames)
    for fname in Covernames[:int(len(Covernames)/2)]:
        src = os.path.join(base_dir, 'Cover', fname)
        dst = os.path.join(train_cover_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to validation_cover_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in Covernames[int(len(Covernames)/2):int(len(Covernames)*0.75)]:
        src = os.path.join(base_dir, 'Cover', fname)
        dst = os.path.join(validation_cover_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to test_cover_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in Covernames[int(len(Covernames)*0.75):]:
        src = os.path.join(base_dir, 'Cover', fname)
        dst = os.path.join(test_cover_dir, fname)
        shutil.copyfile(src, dst)
       
    # Copy first 1000 dog images to train_dogs_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    Backgroundnames = os.listdir(os.path.join(base_dir, 'Background'))
    random.shuffle(Backgroundnames)
    for fname in Backgroundnames[:int(len(Backgroundnames)/2)]:
        src = os.path.join(base_dir, 'Background', fname)
        dst = os.path.join(train_background_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to validation_background_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in Backgroundnames[int(len(Backgroundnames)/2):int(len(Backgroundnames)*0.75)]:
        src = os.path.join(base_dir, 'Background', fname)
        dst = os.path.join(validation_background_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to test_background_dir
    #fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in Backgroundnames[int(len(Backgroundnames)*0.75):]:
        src = os.path.join(base_dir, 'Background', fname)
        dst = os.path.join(test_background_dir, fname)
        shutil.copyfile(src, dst)
    return

#building the simnple model 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dense(4, activation='softmax'))

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=(50, 50),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(50, 50),
    batch_size=20,
    class_mode='categorical')

model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


"""
train_data_path = r'E:\400 Data analysis\410 Plant count\Training_data'

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
                    #global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
                    #Normalize The feature vectors...
                    X.append(image)
                    #check wether to use strings or integers
                    y_train.append(str(folder))
    
    X = np.asarray(X)     
    y_train = np.asarray(y_train)
    #X = X[:,1:]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    toc = time.time()
    processing_time = toc - tic
    print('Importing training data took ' + str(processing_time) + ' seconds')
    return X, y_train, scaler



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
"""
