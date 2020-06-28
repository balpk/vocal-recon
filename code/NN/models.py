import numpy as np
import tensorflow as tf
from tensorflow import keras


def AutoEnc(input_shape = (81,243,1)):

    input = keras.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(64, kernel_size=(1, 3),strides=(1,3))(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(128, kernel_size=(4, 4),strides=(2,2))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(256, kernel_size=(4, 4),strides=(2,2))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(512, kernel_size=(4, 4),strides=(2,2))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(128, kernel_size=(2, 2),strides=(8,8))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2DTranspose(512, kernel_size=(2,2),strides=(8,8))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4,4),strides=(2,2))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(128, kernel_size=(4,4),strides=(2,2))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(64, kernel_size=(4,4),strides=(2,2))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(1, kernel_size=(4,4),strides=(1,1))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(1, kernel_size=(1,3),strides=(1,3))(x)
    x = keras.layers.ELU()(x)
    
    model = keras.models.Model(input, x)
    return model

def CNN(input_shape = (80,240,2)):

    input = keras.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(16, kernel_size=(1, 3),strides=(1,3))(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(32, kernel_size=(4, 4),strides=(2,2),padding = 'same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(64, kernel_size=(4, 4),strides=(2,2),padding = 'same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2DTranspose(32, kernel_size=(4,4),strides=(2,2),padding = 'same')(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(16, kernel_size=(4,4),strides=(2,2),padding = 'same')(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(1, kernel_size=(1,3),strides=(1,3),padding = 'same')(x)
    x = keras.layers.ELU()(x)
    
    model = keras.models.Model(input, x)
    return model

def FC(input_shape = (81,243,1)):

    input = keras.Input(shape=input_shape)
    x = keras.layers.Reshape((81*243*1,))(input)
    
    x = keras.layers.Dense(256,activation='relu')(x)
    x = keras.layers.Dense(81*243, activation='sigmoid')(x)
    x = keras.layers.Reshape((81,243,1))(x)

    
    model = keras.models.Model(input, x)
    return model

def FC_t_trivial(input_shape = (8000,)):

    input = keras.Input(shape=input_shape)
    
    x = keras.layers.Dense(8000,activation='linear')(input)
    
    model = keras.models.Model(input, x)
    return model