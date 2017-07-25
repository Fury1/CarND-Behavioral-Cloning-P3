# Project 3 Behavioral Cloning

import csv
import os
import cv2
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sys import argv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint


def model_training(new_model=True, model=None):
    """
    Model_training() allows a new model to be trained with the below architecture, or a previous model
    checkpoint to be loaded and re-trained with the below architecture.

    new_model = Boolean
    trained_model = [optional] string --> path to a trained model

    Model training default is a new model.
    If new_model is set to False, a trained model argument must be supplied.
    """

    if new_model:
        # NVIDIA Model architecture
        model = Sequential()
        model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3))) # crop the images to get rid of irrelevant features
        model.add(Lambda(lambda x: (x - 128) / 128)) # normalize all pixels to a mean of 0 +-1
        model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid', activation='elu')) # 1st convolution
        model.add(BatchNormalization())
        model.add(Conv2D(36, (5,5), strides=(2,2), padding='valid', activation='elu')) # 2nd convolution
        model.add(BatchNormalization())
        model.add(Conv2D(48, (5,5), strides=(2,2), padding='valid', activation='elu')) # 3rd convolution
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='elu')) # 4th convolution
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='elu')) # 4th convolution
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Flatten()) # flatten the dimensions
        model.add(Dense(100, activation='elu')) # 1st fully connected layer
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='elu')) # 2nd fully connected layer
        model.add(BatchNormalization())
        model.add(Dense(10, activation='elu')) # 3rd fully connected layer
        model.add(Dense(1)) # model prediction output node

    if not new_model:
        # Reloads a previously trained model to allow for re-training on added data
        model = load_model(model)

    return model


# Create a generator to limit memory usage and feed the model data generated batches
def batch_generator(samples, batch_size):

    global pickled_steering_angles

    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        # Shuffle the selected samples
        shuffle(samples)

        # Proccess the samples batch by batch
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # Setup lists to keep track of images and coresponding steering angles
            images = []
            steering_angles = []

            for batch_sample in batch_samples:
                # Data augmentation, utilize three images per frame
                # Based on the image selected add a steering angle correction to aid in recovery data
                correction = 0.125
                steering_angle = float(batch_sample[3])  # batch_sample[3] are the steering angles
                steering_angle_left = steering_angle + correction
                steering_angle_right = steering_angle - correction

                # Get image filenames
                filename_center = re.split(r'[\\/]+', batch_sample[0])[-1] # extract non windows & windows file paths
                filename_left = re.split(r'[\\/]+', batch_sample[1])[-1]
                filename_right = re.split(r'[\\/]+', batch_sample[2])[-1]

                # Read in the center, left, and right images
                image_center = cv2.imread('data\\data\\IMG\\' + filename_center)
                image_left = cv2.imread('data\\data\\IMG\\' + filename_left)
                image_right = cv2.imread('data\\data\\IMG\\' + filename_right)

                # Drop a percentage of 0 degree steering angle values (90%)
                drop_probability = np.random.random()

                if steering_angle == 0 and drop_probability > 0.1:
                    pass

                else:
                    # Randomize image flips to further augment data, then add to respective lists
                    flip_probabilty = np.random.random()

                    if flip_probabilty > 0.5:
                        images.extend([np.fliplr(image_center), np.fliplr(image_left), np.fliplr(image_right)])
                        steering_angles.extend([steering_angle * -1.0, steering_angle_left * -1.0, steering_angle_right * -1.0])
                    else:
                        images.extend([image_center, image_left, image_right])
                        steering_angles.extend([steering_angle, steering_angle_left, steering_angle_right])

            # Convert the python lists to numpy arrays arrays for Keras
            X_train = np.array(images)
            y_train = np.array(steering_angles)

            # Add the batch of finalized steering angles to the list for pickle file
            pickled_steering_angles.extend(y_train)

            yield (X_train, y_train)


# Set up a list for the lines to be added to so they can be manipulated
# Create another list to collect steering angles from the batch generated for pickle file
samples = []
pickled_steering_angles = []

# Add in the lines from the csv file to a list
with open('data\\data\\driving_log.csv') as data_file:
    reader = csv.reader(data_file)
    next(reader) # skip the column header row in the file
    for line in reader:
        samples.append(line)

# Split up the validation and training data randomly
training_data, validation_data = train_test_split(samples, test_size = 0.3)

"""
Call the model training method, new or re-train.

New:
    model_training()
Re-train:
    model_training(new_model=False, model='model_to_load')
"""
model = model_training()

# Checkpoint the model during training to find the best performance, this allows the best model to be selected or re-trained (if needed)
filepath = "Checkpoints\\model.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=1)

# Set up the loss and optimizer functions, configure the model
# Mean squared error was chosen for a loss because it is the difference of the estimate vs actual values
Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=Adam)

# Hyperparameters
batch_size = 256
epochs = 25

# Record the Keras history object for plotting
keras_history_object = model.fit_generator(batch_generator(training_data, batch_size),
                                            steps_per_epoch=(len(training_data) * 3) / batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            callbacks=[checkpoint],
                                            validation_data=batch_generator(validation_data, batch_size),
                                            validation_steps=(len(validation_data) * 3) / batch_size)


# Create pickle file of collective steering angles to plot histogram chart later
with open('steering_angles.p', 'wb') as f:
    pickle.dump(pickled_steering_angles, f)

# Plot the training and validation losses for visualization of this run
plt.figure(1, figsize=(25, 15))
plt.subplot(211)
plt.plot(keras_history_object.history['loss'])
plt.plot(keras_history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.subplot(212)

# Histogram plot of steering angle measurment samples for this batch of samples
plt.hist(pickled_steering_angles, bins=50)
plt.title('Model Measurement Distribution')
plt.ylabel('# Samples')
plt.xlabel('Steering Angle')
plt.show()
