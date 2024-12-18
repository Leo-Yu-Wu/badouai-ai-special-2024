import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import Sequential,Model

def alexNet(input_shape=(224,224,3),output_shape=2):
    model = Sequential()
    model.add(Conv2D(
        filters=48,
        kernel_size=(11,11),
        strides=(4,4),
        padding="valid",
        activation="relu",
        input_shape = input_shape
        ))
    model.add(BatchNormalization())
    model.add(MaxPool2D(
        pool_size=(3,3),
        strides=(2,2),
        padding="valid"))
    model.add(Conv2D(
        filters=128,
        kernel_size=(5,5),
        strides=(1,1),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(MaxPool2D(
        pool_size=(3,3),
        strides=(2,2),
        padding="valid"
    ))
    model.add(Conv2D(
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation="relu"
    ))
    model.add(Conv2D(
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation="relu"
    ))
    model.add(Conv2D(
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation="relu"
    ))
    model.add(MaxPool2D(
        pool_size=(3,3),
        strides=(2,2),
        padding="valid"
    ))
    model.add(Flatten())
    model.add(Dense(
        units=1024,
        activation="relu",
    ))
    model.add(Dropout(0.25))
    model.add(Dense(
        units=1024,
        activation="relu",
    ))
    model.add(Dropout(0.25))
    model.add(Dense(
        units = output_shape,
        activation="softmax"
    ))
    return model