from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D
from keras.layers.pooling import MaxPooling3D
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend
from keras.layers.merge import Concatenate
import numpy as np
import tensorflow as tf
from keras.layers.recurrent import GRU
from keras.layers import Input, merge
from keras.models import Model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d, conv_3d, max_pool_3d, avg_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge

def get_3d_model(shape):
    model = Sequential()
    model.add(Conv3D(36, 3, strides=(2, 2, 2), padding="same", activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(Conv3D(48, 3, padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, 3, padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(94, 3, padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='valid'))
    model.add(Flatten())
    model.add(Dense(1524))
    model.add(Activation('relu'))
    model.add(Dense(750))
    model.add(Activation('relu'))
    model.add(Dense(370))
    model.add(Activation('relu'))
    model.add(Dense(180))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
    #backend.get_session().run(tf.initialize_all_variables())
    return model

'''Model as proposed in Bojarski et al. 2016

'''

def get_model(shape):
    model = Sequential()
    model.add(Convolution2D(36, 5, strides=(2, 2), padding="same", input_shape=shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 5, strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, 3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, 3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1524))
    model.add(Activation('relu'))
    model.add(Dense(750))
    model.add(Activation('relu'))
    model.add(Dense(370))
    model.add(Activation('relu'))
    model.add(Dense(180))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
    #backend.get_session().run(tf.initialize_all_variables())
    return model

def get_vis_comb_2(map_shape, cam_shape, speed_shape):
    a = Input(shape=map_shape)
    b = Input(shape=cam_shape)
    c = Input(shape=speed_shape)

    c1 = Convolution2D(64, 5, strides=(2, 2), padding="same", activation ='relu')(b)
    c1 = BatchNormalization()(c1)
    c1 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='relu')(c1)
    c1 = MaxPooling2D(pool_size=(3, 3))(c1)
    c1 = BatchNormalization()(c1)
    c3 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(c1)

    c3 = BatchNormalization()(c3)
    c3 = Convolution2D(256, 3, padding="same", activation ='relu')(c3)
    c3 = BatchNormalization()(c3)

    c5 = Convolution2D(256, 3, padding="same", activation ='relu')(c3)
    c5 = Concatenate(axis=3)([c3, c5])
    c5 = MaxPooling2D(pool_size=(3, 3))(c5)
    c5 = BatchNormalization()(c5)
    flatten1 = Flatten()(c5)

    c1_2 = Convolution2D(64, 5, strides=(1, 1), padding="same", activation ='relu')(a)
    c1_2 = MaxPooling2D(pool_size=(3, 3))(c1_2)
    c1_2 = BatchNormalization()(c1_2)
    c2_2 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(c1_2)
    c1_2 = MaxPooling2D(pool_size=(3, 3))(c1_2)
    c1_2 = BatchNormalization()(c1_2)
    c1_2 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(c1_2)
    c1_2 = MaxPooling2D(pool_size=(2, 2))(c1_2)
    c1_2 = BatchNormalization()(c1_2)
    flatten2 = Flatten()(c1_2)

    c1_3 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='relu')(c)
    c1_3 = MaxPooling2D(pool_size=(3, 3))(c1_3)
    c1_3 = BatchNormalization()(c1_3)
    c1_3 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(c1_3)
    c1_3 = MaxPooling2D(pool_size=(2, 2))(c1_3)
    c1_3 = BatchNormalization()(c1_3)
    flatten3 = Flatten()(c1_3)

    merged_vector = Concatenate(axis=1)([flatten1, flatten2])
    fc1 = Dense(1250, activation ='relu')(merged_vector)
    fc1 = Dense(512, activation ='relu')(fc1)
    fc1 = Dense(2, activation='linear')(fc1)

    merged_vector = Concatenate(axis=1)([flatten1, flatten3])
    fc1_2 = Dense(1250, activation ='relu')(merged_vector)
    fc1_2 = Dense(512, activation ='relu')(fc1_2)
    fc1_2 = Dense(1, activation='linear')(fc1_2)

    output = Concatenate(axis=1)([fc1, fc1_2])

    model = Model(input=[a,b,c], output=output)
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
    return model


def get_vis_comb(map_shape, cam_shape, speed_shape):
    a = Input(shape=map_shape)
    b = Input(shape=cam_shape)
    c = Input(shape=speed_shape)

    c1 = Convolution2D(64, 5, strides=(1, 1), padding="same", activation ='relu')(b)
    m1 = MaxPooling2D(pool_size=(3, 3))(c1)
    n1 = BatchNormalization()(m1)
    c2 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='relu')(n1)
    m2 = MaxPooling2D(pool_size=(3, 3))(c2)
    n2 = BatchNormalization()(m2)
    c3 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(n2)
    m3 = MaxPooling2D(pool_size=(2, 2))(c3)
    n3 = BatchNormalization()(m3)
    c4 = Convolution2D(256, 3, padding="same", activation ='relu')(n3)
    m4 = MaxPooling2D(pool_size=(2,2))(c4)
    n4 = BatchNormalization()(m4)
    c5 = Convolution2D(256, 3, padding="same", activation ='relu')(n4)
    m5 = MaxPooling2D(pool_size=(2, 2))(c5)
    n5 = BatchNormalization()(m5)
    flatten1 = Flatten()(n5)

    c1_2 = Convolution2D(64, 5, strides=(1, 1), padding="same", activation ='relu')(a)
    m1_2 = MaxPooling2D(pool_size=(3, 3))(c1_2)
    n1_2 = BatchNormalization()(m1_2)
    c2_2 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(n1_2)
    m2_2 = MaxPooling2D(pool_size=(3, 3))(c2_2)
    n2_2 = BatchNormalization()(m2_2)
    c3_2 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(n2_2)
    m3_2 = MaxPooling2D(pool_size=(2, 2))(c3_2)
    n3_2 = BatchNormalization()(m3_2)
    flatten2 = Flatten()(n3_2)

    c1_3 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='relu')(c)
    m1_3 = MaxPooling2D(pool_size=(3, 3))(c1_3)
    n1_3 = BatchNormalization()(m1_3)
    c2_3 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(n1_3)
    m2_3 = MaxPooling2D(pool_size=(2, 2))(c2_3)
    n2_3 = BatchNormalization()(m2_3)
    flatten3 = Flatten()(n2_3)

    merged_vector = Concatenate(axis=1)([flatten1, flatten2, flatten3])
    fc1 = Dense(1250, activation ='relu')(merged_vector)
    fc2 = Dense(1250, activation ='relu')(fc1)
    fc3 = Dense(1250, activation ='relu')(fc2)
    fc3 = Dense(6, activation='linear')(fc2)
    model = Model(input=[a,b,c], output=fc3)
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
    return model

def steering_model(cam_shape, speed_shape):
    b = Input(shape=cam_shape)
    c = Input(shape=speed_shape)

    c1 = Convolution2D(64, 5, strides=(2, 2), padding="same", activation ='relu')(b)
    c1 = BatchNormalization()(c1)
    c1 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='relu')(c1)
    c1 = MaxPooling2D(pool_size=(3, 3))(c1)
    c1 = BatchNormalization()(c1)
    c3 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(c1)

    c3 = BatchNormalization()(c3)
    c3 = Convolution2D(256, 3, padding="same", activation ='relu')(c3)
    c3 = BatchNormalization()(c3)

    c5 = Convolution2D(256, 3, padding="same", activation ='relu')(c3)
    c5 = Concatenate(axis=3)([c3, c5])
    c5 = MaxPooling2D(pool_size=(3, 3))(c5)
    c5 = BatchNormalization()(c5)
    flatten1 = Flatten()(c5)

    c1_3 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='relu')(c)
    c1_3 = MaxPooling2D(pool_size=(3, 3))(c1_3)
    c1_3 = BatchNormalization()(c1_3)
    c1_3 = Convolution2D(128, 3, strides=(1, 1), padding="same", activation ='relu')(c1_3)
    c1_3 = MaxPooling2D(pool_size=(2, 2))(c1_3)
    c1_3 = BatchNormalization()(c1_3)
    flatten3 = Flatten()(c1_3)

    merged_vector = Concatenate(axis=1)([flatten1, flatten3])
    fc1 = Dense(1250, activation ='relu')(merged_vector)
    fc1 = Dense(1250, activation ='relu')(fc1)
    fc1 = Dense(2, activation='linear')(fc1)

    model = Model(input=[b,c], output=fc1)
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
    return model
