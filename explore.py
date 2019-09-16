#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:18:54 2019

@author: pedro
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils import j, train_path, test_path, train_resampled_path

df_tr = pd.read_csv(j(train_path(), 'train.csv'), nrows=150000)
# too much data crushes my home laptop
#df_tr = utils.read_file(j(train_path(), 'train.csv'))
df_te = pd.read_csv(j(test_path(), 'seg_0a0fbb.csv'))

#df_tr.plot()

t = df_tr.time_to_failure
#t.plot()
freq = np.abs(1./np.mean(t[0:100].diff()))
dt = df_tr.time_to_failure.diff()
dt[0] = dt[1]
#dt.plot()

# weird jumps
weirdjumps = t[dt < 10*dt[0:100].mean()]
freq_jumps = np.abs(1./np.mean(weirdjumps.diff()))
weirdjumps.plot()
# use them to guess the frequencies

#%%
# Resample the thing
chunk_l =[]
# read in chunks
df_chunks = pd.read_csv(j(train_path(), 'train.csv'),
                        chunksize=100000)
for chunk in df_chunks:
    dt_c = chunk.time_to_failure.diff()
    chunk_filter = chunk.loc[dt_c < 5*dt_c[0:100].mean(), :]
    chunk_l.append(chunk_filter)

df_resampled = pd.concat(chunk_l)
df_resampled = df_resampled.reset_index()
df_first = df_resampled.loc[2000:15000,:].copy()
#df_first.loc[:, ['acoustic_data']].plot()

f,t,Sxx = signal.spectrogram(df_first.loc[:, ['acoustic_data']].T.values[0], freq_jumps)
#plt.pcolormesh(t, f, Sxx)
#plt.show()

#df_resampled.to_csv(j(train_resampled_path(), 'train_resampled.csv')) # INDEX IS FALSE

#%% 

df = pd.read_csv(j(train_resampled_path(), 'train_resampled.csv'), index_col=0)

#%% Test the feature engineering

import pandas as pd
from utils import j, train_path, test_path, train_resampled_path

df = pd.read_csv(j(train_path(), 'train_event12.csv'))#, nrows=1e6)

data = df.acoustic_data#.values
t = df.time_to_failure#.values

#t.diff().mean()*1500*50
#-0.019481291423263825

#%% PSD

from scipy.signal import welch, spectrogram
import matplotlib.pyplot as plt
import numpy as np

freq = 909090893.1880873
f, S = welch(data[100:251], freq, nperseg=151)
plt.semilogy(f,S)
#f,ts,Sxx = spectrogram(data, freq, nperseg=180, noverlap=179)
#plt.pcolormesh(ts, f, np.log(Sxx))
#plt.savefig('pics/psd_event12.pdf')

# spectrogram based on 180 samples and default options, keep Sxx
# corresponding to frequencies f < 1.1e8

# spectrogram made in chunks of 4096 samples and interpolated. why are 179
# samples missing??

# Use same window style for statistical features and S and E

#%%

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

preds = np.load('preds_lstm.npy')
target = load('y_train_1500.joblib')

df = pd.DataFrame(preds).rolling(100).apply(lambda x: 60000*(np.mean(x)-0.3511))
plt.plot(df)
plt.plot(target)


#%% filter

from scipy.signal import iirdesign, filtfilt, convolve
import pandas as pd

#### FILTER STARTS HERE
freq = 909090893.1880873
nyq = freq/2
wp = 1.1e8/nyq
ws = 0.3
b, a = iirdesign(wp, ws, 1, 40)
filt = filtfilt(b, a, data)
# filt is the filtered data
#### FILTER ENDS HERE

import scipy.signal.windows as win
#w = win.tukey(7)
w = win.hanning(7)
w = w/sum(w)

conv = convolve(w, data)
conv = conv[3:]

data.plot()
pd.Series(filt).plot()
pd.Series(conv).plot()

# for E, S, convolve with hanning window of size 7, shift signal 3 samples to
# the left

#%% Plots

df.acoustic_data.plot()
df.time_to_failure.plot()

#%% Energy over a window

from feature_engineering import signal_energy
import matplotlib.pyplot as plt

energy = df.acoustic_data.rolling(150000).apply(signal_energy, raw=False)
energy.plot(logy=True)
plt.savefig('myfile.pdf')

#%% Entropy over a window

from feature_engineering import signal_svd_entropy
import matplotlib.pyplot as plt

entropy = df.acoustic_data.rolling(150000).apply(signal_svd_entropy, raw=False)
entropy.plot()
plt.savefig('myfile.pdf')


#%%


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# Butterworth testing
N = 3
Wd = [0.2]
b, a = signal.butter(N, Wd)
filtered = signal.lfilter(b, a, data)

# Gaussian testing
gaussian_window = signal.windows.gaussian(100, std=2)
gaussian_window = gaussian_window/np.sum(gaussian_window)
filtered_g = signal.convolve(data, gaussian_window)
    
# Comparisons
von=int(10e5)
bis=int(15e5)
plt.subplot(1,2,1)
plt.plot(data[von:bis])
plt.plot(filtered[von:bis])
plt.subplot(1,2,2)
plt.plot(data[von:bis])
plt.plot(filtered_g[von:bis])
#plt.subplot(1,2,2)
#plt.plot(t[von:bis])
plt.show()

#%%

# Noise does not seem to affect much
import matplotlib.pyplot as plt
from feature_engineering import accumulate_signal_energy, accumulate_signal_energy_butter, accumulate_signal_energy_gaussian

energy = accumulate_signal_energy(data)
energy_butter = accumulate_signal_energy_butter(data)
energy_gaussian = accumulate_signal_energy_gaussian(data)

plt.subplot(1,3,1)
plt.plot(energy)
plt.subplot(1,3,2)
plt.plot(energy_butter)
plt.subplot(1,3,3)
plt.plot(energy_gaussian)
plt.show()

#%%

# View energy jumps more clearly
import matplotlib.pyplot as plt
from feature_engineering import accumulate_signal_energy

energy = accumulate_signal_energy(data)
plt.subplot(1,2,1)
plt.plot(energy)
plt.subplot(1,2,2)
plt.plot(data)

#%% Calculate energy of all segments
import pandas as pd
from utils import j, train_path
from feature_engineering import accumulate_signal_energy

for ii in range(18):
    df = pd.read_csv(j(train_path(), 'train_event{:02d}.csv'.format(ii)))
    data = df.acoustic_data.values
    energy = accumulate_signal_energy(data)
    print('Event {:02d}: {:f}'.format(ii, energy[-1]))

#Event 00: 3222358874.000000
#Event 01: 6180440033.000000
#Event 02: 6147804041.000000
#Event 03: 4657237774.000000
#Event 04: 6517884551.000000
#Event 05: 4321987116.000000
#Event 06: 4579348167.000000
#Event 07: 6438971596.000000
#Event 08: 4465278037.000000
#Event 09: 5604304735.000000
#Event 10: 5514804248.000000
#Event 11: 5297438587.000000
#Event 12: 5701799564.000000
#Event 13: 4259781891.000000
#Event 14: 6754155473.000000
#Event 15: 5435432481.000000
