import os
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import sys

nums = [9,11,12,14,15,18,19,20,27,28,29,30,31,33,43,44,45,46,47,48,49,50,51,52,53]

for i in nums:
	data = np.load('/Users/luc/Documents/ETH/NSys_Project/data/rec_'+format(i, '02d')+'_mic_s.npy')
	if i == 9:
		data_stack = data
	else:
		data_stack = np.concatenate((data_stack,data), axis=0)

	label = pd.read_csv('/Users/luc/Documents/ETH/NSys_Project/data/rec_'+str(i)+'_labels.csv')

	if "Wing_flap" in label:
		is_good = ((label['acc_m vocal'] == 1) | (label['acc_f vocal'] == 1)) & (label['acc_m radio'] == 0) & (label['acc_f radio'] == 0) & (label['acc_m vocal'] != label['acc_f vocal']) & (label['Wing_flap'] == 0)

	else:
		is_good = ((label['acc_m vocal'] == 1) | (label['acc_f vocal'] == 1)) & (label['acc_m radio'] == 0) & (label['acc_f radio'] == 0) & (label['acc_m vocal'] != label['acc_f vocal'])

	if i == 9:
		bool_stack = is_good
	else:
		bool_stack = np.concatenate((bool_stack,is_good), axis=0)

	is_male = ((label['acc_m vocal'] == 1) & (label['acc_f vocal'] == 0))
	if i == 9:
		male_stack = is_male
	else:
		male_stack = np.concatenate((male_stack, is_male), axis=0)

print(np.shape(data_stack))
print(np.shape(bool_stack))
print(len(bool_stack))

s_trivial_male = data_stack[8]
s_trivial_male = np.expand_dims(s_trivial_male, 0)

s_trivial_female = data_stack[0]
s_trivial_female = np.expand_dims(s_trivial_female, 0)

for i in range(len(bool_stack)):
	if (bool_stack[i] == True) & (male_stack[i] == True):
		s_trivial_male = np.concatenate((s_trivial_male, np.expand_dims(data_stack[i], 0)),axis=0)
	elif (bool_stack[i] == True) & (male_stack[i] == False):
		s_trivial_female = np.concatenate((s_trivial_female, np.expand_dims(data_stack[i], 0)),axis=0)


print(np.shape(s_trivial_male))
print(np.shape(s_trivial_female))
np.save('s_trivial_male-mic_s', s_trivial_male)