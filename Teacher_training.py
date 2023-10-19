# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:48:15 2022

@author: hayatu
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
#import keras.backend as K
from data_generator import TrainGenerator, ValGenerator, PredictGenerator
from keras import Model
import math
import networks
import os
import datetime
from distiller import Distiller
from keras.layers.core import Dense, Dropout, Flatten



dataset_frames = np.load('./UCF101/features.npy')
dataset_labels = np.load('./UCF101/labels.npy')

# Split features and labels into training and testing data with 80% training and 20% testing data
train_x, test_x, train_y, test_y = train_test_split(dataset_frames, dataset_labels, test_size = 0.20, shuffle = True)


print("Training Data : ", train_x.shape)
print("Validation Data : ", test_x.shape)


batch_size = 8
Epochs = 50
training_generator = TrainGenerator(train_x, train_y, batch_size)
val_generator = TrainGenerator(test_x, test_y, batch_size)


nb_classes = 101

source_model = networks.C3D(weights='sports1M')
source_model.summary()

# freeze the first 19 layers 
for layer in source_model.layers[:19]:
  layer.trainable = False

for i, layer in enumerate(source_model.layers):
  print(i, layer.name, "-", layer.trainable)

print(source_model.layers[-2].output)

X = source_model.layers[-2].output
predictions = Dense(nb_classes, activation='softmax')(X)
teacher = Model(source_model.input, predictions)
teacher.summary()


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Compile the model and specify loss function, optimizer and metrics values to the model
teacher.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = opt, metrics = ["accuracy"])


model_training_history = teacher.fit(
    training_generator,
    steps_per_epoch=math.floor(len(train_x) // batch_size),
    validation_data=val_generator,
    validation_steps=math.floor(len(test_x) // batch_size),
    verbose=1,
    epochs=Epochs,
)
test_history = teacher.evaluate(val_generator)

teacher.save('./Fintuned_teacher_with_pretrained_C3D_Sports1M_weights.h5')
np.save('./Fintuned_teacher_with_pretrained_C3D_Sports1M_training_history',model_training_history.history)




