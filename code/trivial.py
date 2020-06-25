import os
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import sys

nums = [9,11,12,14,15,18,19,20,27,28,29,30,31,33,43,44,45,46,47,48,49,50]

for i in nums:
	data = np.load('/Users/luc/Documents/ETH/NSys_Project/data/rec_'+format(i, '02d')+'_acc_f_s.npy')
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

print(np.shape(data_stack))
print(np.shape(bool_stack))

for i in range(len(bool_stack)):
	if i == 0:
		if bool_stack[i] == True:
			s_trivial = data_stack[i]
			s_trivial = np.expand_dims(s_trivial, 0)
	else:
		if bool_stack[i] == True:
			s_trivial = np.concatenate((s_trivial, np.expand_dims(data_stack[i], 0)),axis=0)

print(np.shape(s_trivial))

