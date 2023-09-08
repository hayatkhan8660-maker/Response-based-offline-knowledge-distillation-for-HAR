import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
from data_generator import TrainGenerator, ValGenerator, PredictGenerator
from keras import Model
import math
import networks
import os
import datetime
from distiller import Distiller
from keras.layers.core import Dense, Dropout, Flatten

dataset_frames = np.load('./Frames_dataset/UCF101/features.npy')
dataset_labels = np.load('./Frames_dataset/UCF101/labels.npy')

# Split features and labels into training and testing data with 80% training and 20% testing data
train_x, test_x, train_y, test_y = train_test_split(dataset_frames, dataset_labels, test_size = 0.20, shuffle = True)


print("Training Data : ", train_x.shape)
print("Validation Data : ", test_x.shape)


batch_size = 8
Epochs = 50
training_generator = TrainGenerator(train_x, train_y, batch_size)
val_generator = TrainGenerator(test_x, test_y, batch_size)

nb_classes = 101

# Loading finetuned teacher 3DCNN model, pre-trained on sports1M dataset
pre_trained_teacher = keras.models.load_model('./Source_pretrained_teacher_finetuned_with_sports1M_weights.h5') 
pre_trained_teacher.summary()

# Loading student model to be supervised by pre_trained_teacher during knowledge distillation
student_model = networks.Student_C3D(nb_classes)
student_model.summary()

# Initialize and compile distiller
distiller = Distiller(student=student_model, teacher=pre_trained_teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=20)

distillation_history = distiller.fit(training_generator, epochs=100)
distiller.evaluate(val_generator)

student_model.save('./Student_with_KD_under_Teacher_with_Finedtuned_C3D_Sports1M.h5')
np.save('./Student_with_KD_under_Teacher_with_Finedtuned_C3D_Sports1M_training_history',distillation_history.history)