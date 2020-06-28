import os
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import sys

acc_m = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_male-acc_s.npy')
mic_m = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_male-mic_s.npy')
acc_f = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_female-acc_s.npy')
mic_f = np.load('/Users/luc/Documents/ETH/NSys_Project/s_trivial_female-mic_s.npy')

fs = 24000
window_size = 256
overlap = 0.875
noverlap = int(np.floor(window_size * overlap))

no = 43

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 10))
f, t, Sxx = signal.spectrogram(acc_m[169], fs)

axs[0].pcolormesh(t, f, np.log10(1e-8 + Sxx))
axs[1].pcolormesh(t, f, np.log10(1e-8 + Sxx))
axs[0].set_ylabel('mic')
axs[1].set_ylabel('acc_m')

plt.show()