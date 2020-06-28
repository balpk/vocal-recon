import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

acc_m = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_male-acc_s.npy')
mic_m = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_male-mic_s.npy')
acc_f = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_female-acc_s.npy')
mic_f = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_female-mic_s.npy')

num = np.arange(800)

train = np.log10(1e-8 + acc_f[num,:,:])
train_scaled = (train - np.amin(train))/(np.amax(train)-np.amin(train))
test = np.log10(1e-8 + acc_f[825:829,:,:])
test_scaled = (test - np.amin(test))/(np.amax(test)-np.amin(test))
y_train = np.log10(1e-8 + mic_f[num,:,:])
y_train_scaled = (y_train - np.amin(y_train))/(np.amax(y_train)-np.amin(y_train))
y_test = np.log10(1e-8 + mic_f[825:829,:,:]) #used to be 295:299 for m
y_test_scaled = (y_test - np.amin(test))/(np.amax(y_test)-np.amin(y_test))

train = train.reshape((800,81*243))
test = test.reshape((4,81*243))
y_train = y_train.reshape((800,81*243))
train_scaled = train_scaled.reshape((800,81*243))
test_scaled = test_scaled.reshape((4,81*243))
y_train_scaled = y_train_scaled.reshape((800,81*243))

reg = LinearRegression()
reg.fit(train_scaled,y_train_scaled)
pred = reg.predict(test_scaled)
pred = pred.reshape((4,81,243))
test_scaled = test_scaled.reshape((4,81,243))
y_test_scaled = y_test_scaled.reshape((4,81,243))
print(np.amax(pred), np.amax(test_scaled), np.amax(y_test_scaled))

idx = 0
for i in range(1,13):
	if i % 3 == 1:
		plt.subplot(4,3,i)
		plt.imshow(test_scaled[idx,:,:])
	elif i % 3 == 2:
		plt.subplot(4,3,i)
		plt.imshow(pred[idx,:,:])
	else:
		plt.subplot(4,3,i)
		plt.imshow(y_test_scaled[idx,:,:])
		idx = idx + 1

plt.savefig('acc_f_trivial_reg_scaled')
plt.show()

np.save('trivial-regression-f-scaled.npy', pred)