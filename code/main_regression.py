import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
#import sounddevice as sd
from sklearn.preprocessing import MinMaxScaler


from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

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

all_df = pd.read_csv(f'data_all/rec_3_labels.csv')
for recording in recordings:
    if recording == 3:
        continue
    string = f'rec_{recording}_labels.csv'
    df = pd.read_csv(f'data_all/{string}')
    all_df = pd.concat((all_df[['Unnamed: 0','Recording','Sequence','acc_m vocal','acc_f vocal']], \
                        df[['Unnamed: 0','Recording','Sequence','acc_m vocal','acc_f vocal']]),axis = 0)
    
which = all_df['acc_m vocal'].to_numpy().astype(int)
num = np.arange(1446)

train = np.log10(1e-8 + S_clean_spec[which,num,:,:])
y = np.log10(1e-8 + S_clean_spec[2,:,:,:])

train = train.reshape((1446,81*243))
y = y.reshape((1446,81*243))

reg = LinearRegression()
reg.fit(train,y)
pred = reg.predict(train)
pred = pred.reshape((1446,81,243))

plt.imshow(pred[502,:,:,2])
plt.show()

np.save('regression.npy',pred)