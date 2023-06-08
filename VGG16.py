# VGG16 model trained on the hybrid dataset

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.regularizers import l2

# Load the dataframes
train_path = '/kaggle/input/hybrid/HybridTrain'
validation_path = '/kaggle/input/hybrid/HybridVal'
test_path = '/kaggle/input/hybrid/HybridTest'

# Initialize the ImageDataGenerator for training data
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Initialize the ImageDataGenerator for test and validation data
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

# Generate the training data
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    classes=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'] 
)

# Generate the validation data
validation_data = test_datagen.flow_from_directory(
    validation_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    classes=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'] 
)

# Generate the test data
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'] 
)

# Load VGG16 without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Train all layers
for layer in base_model.layers:
    layer.trainable = True

# Add the custom layers...!!
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mdlcheckpoint_cb = ModelCheckpoint('/kaggle/working/VGG_HybridDS.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(train_data, 
                    epochs=100, 
                    validation_data=validation_data, 
                    callbacks=[early_stopping_cb, mdlcheckpoint_cb, reduce_lr])

# Test accuracy for hybrid test set
model.evaluate(test_data)

test_highres_path = '/kaggle/input/hybrid/TestSets/HighResTest'
test_lowres_path = '/kaggle/input/hybrid/TestSets/LowResTest'

# Generate the test data for high res ONLY
test_highres_data = test_datagen.flow_from_directory(
    test_highres_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'] 
)

# Generate the test data for low res ONLY
test_lowres_data = test_datagen.flow_from_directory(
    test_lowres_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'] 
)

# Test accuracy for high res ONLY test set
model.evaluate(test_highres_data)
# Test accuracy for low res ONLY test set
model.evaluate(test_lowres_data)