import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# A solution for a problem (found online)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

recordings = [3,5,6,9,10,11,12,14,15,17,18,19,20,27,28,29,30,31,33,43,44,45,46,47,48,49,50,51,52,53]
substrings_spec = ['acc_f_s', 'acc_m_s', 'mic_s']

S_clean_spec = np.zeros((3,1446,81,243))
ind = 0
for recording in recordings:
    for idx,substring in enumerate(substrings_spec):
        string = f'rec_{recording:02}_{substring}.npy'
        test = np.load(f'data_all/{string}')
        if np.size(test) != 0:
            seq = test.shape[0]
            S_clean_spec[idx,ind:ind+seq] = test
    if np.size(test) != 0:
        ind = ind + seq
        
S_clean_ch = np.rollaxis(S_clean_spec, 0, 4) 

def AutoEnc(input_shape = (81,243,2)):

    input = keras.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(8, kernel_size=(1, 3),strides=(1,3))(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(8, kernel_size=(4, 4),strides=(2,2))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(4, kernel_size=(4, 4),strides=(2,2))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(8, kernel_size=(4, 4),strides=(2,2))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2D(16, kernel_size=(2, 2),strides=(8,8))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Conv2DTranspose(16, kernel_size=(2,2),strides=(8,8))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(8, kernel_size=(4,4),strides=(2,2))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(4, kernel_size=(4,4),strides=(2,2))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(8, kernel_size=(4,4),strides=(2,2))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Conv2DTranspose(8, kernel_size=(4,4),strides=(1,1))(x)
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

model = CNN()
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)

log = np.log10(1e-8 + S_clean_ch[:,:80,:240,:])
scaled = (log - np.amin(log))/(np.amax(log)-np.amin(log))

model.fit(scaled[:,:,:,1:3],np.expand_dims(scaled[:,:,:,0],axis=3), \
         epochs = 10,shuffle = True,batch_size = 32)

pred = model.predict(S_clean_ch[:,:,:,1:3])
np.save('pred.npy',pred)