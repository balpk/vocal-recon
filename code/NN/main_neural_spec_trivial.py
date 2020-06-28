# Might comment out the lines 11-14, it was a solution to a problem I faced, you might don't have the same problem.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# A solution for a problem (found online)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import models

folder = 'datasets/'

Data = np.load(f'{folder}S_trivial_spec_female_ch_last.npy')
Data = np.log10(1e-8 + Data)

#Data[:,:,0] = scale(Data[:,:,0], axis =1)
#Data[:,:,1] = scale(Data[:,:,1], axis =1)

# first index is 0 for acc data of female, second one is 1 for mic data. This is only for S_trivial, if you will take S_clean
# then the index 0 is female, index 1 is male, index 2 is mic data.
X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(Data[:,:,:,0],axis=-1),\
                                                    np.expand_dims(Data[:,:,:,1],axis=-1), test_size = 0.2, random_state = 7)

y_test = (y_test - np.amin(y_train))/(np.amax(y_train)-np.amin(y_train))
y_train = (y_train - np.amin(y_train))/(np.amax(y_train)-np.amin(y_train))

X_test = (X_test - np.amin(X_train))/(np.amax(X_train)-np.amin(X_train))
X_train = (X_train - np.amin(X_train))/(np.amax(X_train)-np.amin(X_train))


# See some examples

plt.figure()
plt.imshow(X_train[0,:,:,0])
plt.figure()
plt.imshow(y_train[0,:,:,0])

plt.figure()
plt.imshow(X_train[1,:,:,0])
plt.figure()
plt.imshow(y_train[1,:,:,0])

plt.figure()
plt.imshow(X_train[2,:,:,0])
plt.figure()
plt.imshow(y_train[2,:,:,0])

plt.show()

model = models.AutoEnc()
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='mean_squared_error', optimizer=opt)

model.fit(X_train,y_train, \
         epochs = 30,shuffle = True,batch_size = 32)

pred = model.predict(X_test)
pred = np.reshape(pred,(-1,81,243))

np.save('pred_trivial_female_spec_nn.npy',pred)


# See examples of results

plt.figure()
plt.imshow(y_test[0,:,:,0])
plt.figure()
plt.imshow(pred[0])

plt.figure()
plt.imshow(y_test[1,:,:,0])
plt.figure()
plt.imshow(pred[1])

plt.figure()
plt.imshow(y_test[2,:,:,0])
plt.figure()
plt.imshow(pred[2])

plt.show()