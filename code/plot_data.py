# This plot can be used for either plotting the triplets for hand labelling,
# or for generating the data from the manual labels, imported as csv files.

# To run, change directory names, comment/uncomment lines exclusive for data
# generation or plot generation.

# As is, the script is in data generation mode. Avoid running the script on
# recordings for which hand labels do not exist. The data may be overwritten.
# In that case, just supply the recordings list instead of assigning the 
# dictionary files.keys() . 

import os
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

files = {
3:'2018-08-14/b8p2male-b10o15female_3_SdrChannels.w64',
5:'2018-08-14/b8p2male-b10o15female_5_SdrChannels.w64',
6:'2018-08-14/b8p2male-b10o15female_6_SdrChannels.w64',
8:'2018-08-14/b8p2male-b10o15female_8_SdrChannels.w64',
9:'2018-08-14/b8p2male-b10o15female_9_SdrChannels.w64',
10:'2018-08-14/b8p2male-b10o15female_10_SdrChannels.w64',
11:'2018-08-14/b8p2male-b10o15female_11_SdrChannels.w64',
12:'2018-08-14/b8p2male-b10o15female_12_SdrChannels.w64',
14:'2018-08-14/b8p2male-b10o15female_14_SdrChannels.w64',
15:'2018-08-14/b8p2male-b10o15female_15_SdrChannels.w64',
17:'2018-08-14/b8p2male-b10o15female_17_SdrChannels.w64',
18:'2018-08-14/b8p2male-b10o15female_18_SdrChannels.w64',
19:'2018-08-14/b8p2male-b10o15female_19_SdrChannels.w64',
20:'2018-08-15/b8p2male-b10o15female_20_SdrChannels.w64',
27:'2018-08-15/b8p2male-b10o15female_27_SdrChannels.w64',
28:'2018-08-16/b8p2male-b10o15female_28_SdrChannels.w64',
29:'2018-08-16/b8p2male-b10o15female_29_SdrChannels.w64',
30:'2018-08-16/b8p2male-b10o15female_30_SdrChannels.w64',
31:'2018-08-16/b8p2male-b10o15female_31_SdrChannels.w64',
31:'2018-08-16/b8p2male-b10o15female_32_SdrChannels.w64',
33:'2018-08-16/b8p2male-b10o15female_33_SdrChannels.w64',
43:'2018-08-18/b8p2male-b10o15female_43_SdrChannels.w64',
44:'2018-08-18/b8p2male-b10o15female_44_SdrChannels.w64',
45:'2018-08-18/b8p2male-b10o15female_45_SdrChannels.w64',
46:'2018-08-18/b8p2male-b10o15female_46_SdrChannels.w64',
47:'2018-08-18/b8p2male-b10o15female_47_SdrChannels.w64',
48:'2018-08-19/b8p2male-b10o15female_48_SdrChannels.w64',
49:'2018-08-19/b8p2male-b10o15female_49_SdrChannels.w64',
50:'2018-08-19/b8p2male-b10o15female_50_SdrChannels.w64',
51:'2018-08-19/b8p2male-b10o15female_51_SdrChannels.w64',
52:'2018-08-19/b8p2male-b10o15female_52_SdrChannels.w64',
53:'2018-08-19/b8p2male-b10o15female_53_SdrChannels.w64'
}


########################################################
# PARAMETERS                                           #
########################################################

# Sampling Rate
fs = 24000 
# For Spectograms
window_size = 256
overlap = 0.875
noverlap = int(np.floor(window_size * overlap))
# The Recordings to use
recordings = [51,52,53]# files.keys() # Can also be a list

# Data generation
# when saving: adjust filename
all_labels = pd.read_csv('manual_label_susanne.csv')
# Filter data that has vocalisations, does not have radio noise
#is_good = ((all_labels['Undecided'] == 0) & (all_labels['Wing_flap'] == 0) & \
#          ((all_labels['acc_m vocal'] == 1) | (all_labels['acc_f vocal'] == 1)) & \
#          (all_labels['acc_m radio'] == 0) & (all_labels['acc_f radio'] == 0))
is_good = ((all_labels['acc_m vocal'] == 1) | (all_labels['acc_f vocal'] == 1)) & (all_labels['acc_m radio'] == 0) & (all_labels['acc_f radio'] == 0)
good_labels = all_labels[is_good]

########################################################
# PARAMETERS                                           #
########################################################

def read_recording(no, names=files):
    '''
    Returns Recording in format
    mic, acc_m, acc_f
    '''
    try:
        file = '/home/bukaya/ethz/ns/b8p2male-b10o15female_aligned/' + files[no]
        with open(file, 'rb') as f:  
            Audiodata, _samplerate_bad = sf.read(f)
        mic   = Audiodata[:, 0]
        acc_m = Audiodata[:, 1]
        acc_f = Audiodata[:, 2]
        del Audiodata
        return mic, acc_m, acc_f
    except KeyError:
        print(f'Recording {no} does not exist.\nTry any of: {files.keys()}')
        return 0, 0, 0
    
def get_time(index, samplerate=24000):
    s_idx = index / samplerate
    hours = s_idx // 3600
    minutes = int((s_idx - hours*3600) // 60)
    seconds = (s_idx - hours*3600 - minutes * 60)
    return [minutes, seconds]


# Create directories
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('plot'):
    os.makedirs('plot')

for no in recordings:
    mic, acc_m, acc_f = read_recording(no)

    # Apply bandpass
    sos = signal.butter(10, [15, 8000], 'bp', fs=24000, output='sos')
    filtered_bp = signal.sosfilt(sos, mic)
    acc_m_bp = signal.sosfilt(sos, acc_m)
    acc_f_bp = signal.sosfilt(sos, acc_f)

    # clear memory
    del mic
    del sos

    # Take moving window RMS
    window = 2400
    sq_fil = np.power(filtered_bp,2)
    mean_pd_abs = pd.Series(sq_fil).rolling(window=window).mean().iloc[window-1:].values
    del sq_fil

    # This method of boolean indexing + where is the fastest I could find, much, much faster than using a for loop to check
    # extract time indices where the microphone is louder than some threshold during the moving RMS window
    mask = mean_pd_abs >= 5e-6
    indices = np.where(mask)
    # The moving RMS window is offset by window//2, because the first data point of RMS corresponds to that point
    # also, indices is a tuple, the first value of which is our desired array.
    indices = indices[0] + window //2
    del mean_pd_abs
    del mask

    a = indices
    s = fs//3
    #Get starting indices
    m = [indices[0]]
    for i in np.arange(1,len(indices)):
        # If there is a jump larger than the RMS window, this must be a separate non-silence part
        if(indices[i] >= indices[i-1]+window):
            m.append(indices[i])
    del indices
    print(f'Total {len(m)} Sequences found in Recording {no}.')

    idxs = []
    offset = fs//20
    for i in m:
        # Do not overflow
        if(i+s-offset < filtered_bp.shape[0]):
            # Do not underflow either
            if(i-offset > 0):
                # Check for radio noise in either accelerometer by time series intensity
                if(np.mean(np.power(acc_m[i-offset:i+s-offset],2)) + np.mean(np.power(acc_f[i-offset:i+s-offset],2)) < 0.08):
                    idxs.append(np.arange(i-offset,i+s-offset))

    idxs = np.asarray(idxs)
    print(f'{idxs.shape[0]} Sequences after reduction in Recording {no}.')
    del m
    del acc_m
    del acc_f

    mic_bp_l = np.zeros(idxs.shape)
    acc_m_l = np.zeros(idxs.shape)
    acc_f_l = np.zeros(idxs.shape)

    for i in range(idxs.shape[0]):
        mic_bp_l[i] = filtered_bp[idxs[i]]
        acc_m_l[i]  = acc_m_bp[idxs[i]]
        acc_f_l[i]  = acc_f_bp[idxs[i]]

    '''
    # Plot generation 
    for i in range(mic_bp_l.shape[0]):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 10))
        fig.subplots_adjust(hspace=0.06)

        f, t, Sxx = signal.spectrogram(mic_bp_l[i], fs , window=signal.hamming(window_size, sym=False) , #window=signal.blackman(self.window_size),
                                        nfft=window_size, noverlap=noverlap, scaling="spectrum")
        axs[0].pcolormesh(t, f[:81], np.log10(1e-8 + Sxx[:81,:]))

        f, t, Sxx = signal.spectrogram(acc_m_l[i], fs , window=signal.hamming(window_size, sym=False) , #window=signal.blackman(self.window_size),
                                        nfft=window_size, noverlap=noverlap, scaling="spectrum")
        axs[1].pcolormesh(t, f[:81], np.log10(1e-8 + Sxx[:81,:]))

        f, t, Sxx = signal.spectrogram(acc_f_l[i], fs , window=signal.hamming(window_size, sym=False) , #window=signal.blackman(self.window_size),
                                        nfft=window_size, noverlap=noverlap, scaling="spectrum")
        axs[2].pcolormesh(t, f[:81], np.log10(1e-8 + Sxx[:81,:]))

        axs[0].set_ylabel('mic')
        axs[1].set_ylabel('acc_m')
        axs[2].set_ylabel('acc_f')
        s_min, s_sec = get_time(idxs[i][0])

        axs[0].set_title(f'Rec {no:02}, Seq {i:03}: From {s_min}:{s_sec:0.2f}')

        fig.savefig(f'plot/rec_{no:02}_seq_{i:03}.png',bbox_inches='tight')
        plt.close(fig)
        if(i%10 == 0):
            print(f'Recording {no}: plot {i} of {mic_bp_l.shape[0]}', end="\r")

    print(f'All plots saved for Recording {no}.')
    '''
    # Data generation
    is_rec = good_labels['Recording'] == no
    rec_labels = good_labels[is_rec]
    # Spectograms and time series saving
    size = rec_labels['Sequence'].shape[0]
    mic_bp_t = np.zeros((size,idxs.shape[1]))
    acc_m_t = np.zeros((size,idxs.shape[1]))
    acc_f_t = np.zeros((size,idxs.shape[1]))
    mic_bp_s = [] #np.zeros((size,idxs.shape[1]))
    acc_m_s = [] #np.zeros((size,idxs.shape[1]))
    acc_f_s = [] #np.zeros((size,idxs.shape[1]))

    for j, s in enumerate(rec_labels['Sequence']):

        f, t, Sxx = signal.spectrogram(mic_bp_l[s], fs , window=signal.hamming(window_size, sym=False) , #window=signal.blackman(self.window_size),
                                        nfft=window_size, noverlap=noverlap, scaling="spectrum")
        mic_bp_t[j] = mic_bp_l[s]
        mic_bp_s.append(Sxx[:81,:])
        f, t, Sxx = signal.spectrogram(acc_m_l[s], fs , window=signal.hamming(window_size, sym=False) , #window=signal.blackman(self.window_size),
                                        nfft=window_size, noverlap=noverlap, scaling="spectrum")
        acc_m_t[j] = acc_m_l[s]
        acc_m_s.append(Sxx[:81,:])
        f, t, Sxx = signal.spectrogram(acc_f_l[s], fs , window=signal.hamming(window_size, sym=False) , #window=signal.blackman(self.window_size),
                                        nfft=window_size, noverlap=noverlap, scaling="spectrum")
        acc_f_t[j] = acc_f_l[s]
        acc_f_s.append(Sxx[:81,:])
        if(s%10 == 0):
            print(f'Recording {no}: data {s} of {size}', end="\r")

    mic_bp_s = np.asarray(mic_bp_s)
    acc_m_s = np.asarray(acc_m_s)
    acc_f_s = np.asarray(acc_f_s)

    np.save(f'data/rec_{no:02}_mic_t.npy', mic_bp_t)
    np.save(f'data/rec_{no:02}_mic_s.npy', mic_bp_s)
    np.save(f'data/rec_{no:02}_acc_m_t.npy', acc_m_t)
    np.save(f'data/rec_{no:02}_acc_m_s.npy', acc_m_s)
    np.save(f'data/rec_{no:02}_acc_f_t.npy', acc_f_t)
    np.save(f'data/rec_{no:02}_acc_f_s.npy', acc_f_s)
    rec_labels.to_csv(f'data/rec_{no}_labels.csv')

    print(f'All data saved for Recording {no}.')
    #del idxs
    
    del idxs
