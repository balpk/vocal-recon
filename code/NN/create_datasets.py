# Run this once
# Saves 6 datasets, trivial male and female spectograms and time series data all seperately as 4 different datasets
# and S_clean as time series and as spectograms --> 2 different datasets.

# need to create a folder to save those see line 10
# to understand the format of the datasets see the comments line 16-17, 68-69

import numpy as np
import pandas as pd

# A new folder to save the created datasets
to_save = 'datasets/'

# where your data is stored
data_from = 'data_all/'

recordings = [3,5,6,9,10,11,12,14,15,17,18,19,20,27,28,29,30,31,33,43,44,45,46,47,48,49,50,51,52,53]
substrings_spec = ['acc_f_s', 'acc_m_s', 'mic_s']

# format of a spectogram dataset channels are in the first dimension here but will be saved as the last dimension and refer to
# acc_f, acc_m, mic data in order.
S_clean_spec = np.zeros((3,1446,81,243))

ind = 0
for recording in recordings:
    for idx,substring in enumerate(substrings_spec):
        string = f'rec_{recording:02}_{substring}.npy'
        test = np.load(f'{data_from}{string}')
        if np.size(test) != 0:
            seq = test.shape[0]
            S_clean_spec[idx,ind:ind+seq] = test
    if np.size(test) != 0:
        ind = ind + seq
        
S_clean_spec_ch_last = np.rollaxis(S_clean_spec, 0, 4)
np.save(f'{to_save}S_clean_spec_ch_last.npy', S_clean_spec_ch_last)

S_clean_time = np.zeros((3,1446,8000))
substrings_spec = ['acc_f_t', 'acc_m_t', 'mic_t']
ind = 0
for recording in recordings:
    for idx,substring in enumerate(substrings_spec):
        string = f'rec_{recording:02}_{substring}.npy'
        test = np.load(f'{data_from}{string}')
        if np.size(test) != 0:
            seq = test.shape[0]
            S_clean_time[idx,ind:ind+seq] = test
    if np.size(test) != 0:
        ind = ind + seq
        
S_clean_t = np.rollaxis(S_clean_time, 0, 3)
np.save(f'{to_save}S_clean_time_ch_last.npy', S_clean_t)

all_df = pd.read_csv(f'{data_from}rec_3_labels.csv')
for recording in recordings:
    if recording == 3:
        continue
    string = f'rec_{recording}_labels.csv'
    df = pd.read_csv(f'{data_from}/{string}')
    all_df = pd.concat((all_df[['Unnamed: 0','Recording','Sequence','acc_m vocal','acc_f vocal']], \
                        df[['Unnamed: 0','Recording','Sequence','acc_m vocal','acc_f vocal']]),axis = 0)
    
all_df.insert(0, "Enum", np.arange(1446))

is_good = (all_df['acc_m vocal'] == 1) & (all_df['acc_f vocal'] == 0)
male_df = all_df[is_good]
enum_m = male_df['Enum'].to_numpy().astype(int)
is_good = (all_df['acc_f vocal'] == 1) & (all_df['acc_m vocal'] == 0)
female_df = all_df[is_good]
enum_f = female_df['Enum'].to_numpy().astype(int)

# For S_trivial, only the neccessary information is saved. the first channel is acc data as input the second channel is mic data to be
# predicted.

S_trivial_spec_female_ch_last = np.concatenate((np.expand_dims(S_clean_spec[0,enum_f],axis=-1),np.expand_dims(S_clean_spec[2,enum_f],axis=-1)),axis=-1)
np.save(f'{to_save}S_trivial_spec_female_ch_last.npy', S_trivial_spec_female_ch_last)

S_trivial_spec_male_ch_last = np.concatenate((np.expand_dims(S_clean_spec[1,enum_m],axis=-1),np.expand_dims(S_clean_spec[2,enum_m],axis=-1)),axis=-1)
np.save(f'{to_save}S_trivial_spec_male_ch_last.npy', S_trivial_spec_male_ch_last)

S_trivial_time_female_ch_last = np.concatenate((np.expand_dims(S_clean_time[0,enum_f],axis=-1),np.expand_dims(S_clean_time[2,enum_f],axis=-1)),axis=-1)
np.save(f'{to_save}S_trivial_time_female_ch_last.npy', S_trivial_time_female_ch_last)

S_trivial_time_male_ch_last = np.concatenate((np.expand_dims(S_clean_time[1,enum_m],axis=-1),np.expand_dims(S_clean_time[2,enum_m],axis=-1)),axis=-1)
np.save(f'{to_save}S_trivial_time_male_ch_last.npy', S_trivial_time_male_ch_last)