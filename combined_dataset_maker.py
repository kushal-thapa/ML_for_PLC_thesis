#
# Libraries
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack 
import cmath
from scipy import signal
import glob
import os
import DICE
from sklearn import preprocessing
from librosa.feature import rms, spectral_centroid
plt.style.use('classic')

#-----------------------------------------------------------------------------#
# Function to calculate amplitude envelope
def amplitude_envelope(signal,frame_size,hop_length):
        amplitude_envelope = []
        for i in range(0,len(signal),hop_length):
            current_frame_amplitude_envelope = max(signal[i:i+frame_size])
            amplitude_envelope.append(current_frame_amplitude_envelope)
        return np.array(amplitude_envelope)
    
# Funtions for spectrogram generation 
def enframe(x,S,L,wdw):
    # Divides the time series signal 'x' into multiple frames of length 'L' 
    # with frame overlap (L-S) and applies window 'wdw'
    # Outputs the windowed frames
    w = signal.get_window(wdw,L)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    for t in range(0,nframes):
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
    return(frames)
def stft(frames,N,Fs):
    # Does short term fourier transform of length 'N' to each frame of 'frames' 
    # containing time-series data of sampling frequency 'Fs'
    # Outputs the complex, magnitude, real, imaginary and phase spectra of each frame
    stft_frames = np.array([ fftpack.fft(x,N) for x in frames])
    #freq_axis = np.linspace(0,Fs,N)
    mag_spectra = np.array([ abs(x) for x in stft_frames ])
    real_spectra = np.array([ x.real for x in stft_frames ])
    imag_spectra = np.array([ x.imag for x in stft_frames ])
    phase_spectra = []
    for i in range(0,len(stft_frames)):
        phase_spectra.append([ cmath.phase(x) for x in stft_frames[i] ])
    phase_spectra = np.array(phase_spectra)
    return(stft_frames, mag_spectra, real_spectra, imag_spectra, phase_spectra)
def stft2level(stft_spectra,max_freq_bin):
    # Converts the spectral values to decibels
    # Used for visualization
    max_magnitude = max([ max(x) for x in stft_spectra ])
    min_magnitude = max_magnitude / 50000.0
    for t in range(0,len(stft_spectra)):
        for k in range(0,len(stft_spectra[t])):
            stft_spectra[t][k] /= min_magnitude
            if stft_spectra[t][k] < 1:
                stft_spectra[t][k] = 1
    level_spectra = [ 10*np.log10(x[0:max_freq_bin]) for x in stft_spectra ]
    return(level_spectra)
#-----------------------------------------------------------------------------#
# Extracting features and labels
bit_size = 1000
frame_size = 500
frame_overlap = 250
window = 'hanning'
fft_size = 1024

i = 0
time_series_dataset, feature_based_dataset_comb, mag_spectra_comb, real_imag_spectra_comb, label_comb = ([] for i in range(5))
time = 0
center_freq = [690,750,810,870, 930, 990, 1050, 1110, 1170, 1230, 1290, 1350, 
               1410, 1470, 1530, 1590, 1650, 1710, 1770, 1830, 1890, 1950, 2010] 
#center_freq = [1170]  ##
#amp = [10,20,50,100,250,500,750,1000]   ##

# files
directory = 'C:\\Users\\kusha\\OneDrive - Texas State University\\thesis\\raw data\\freq_analysis_1A'      #
#directory = 'C:\\Users\\kusha\\OneDrive - Texas State University\\thesis\\raw data\\amp_analysis_1170Hz'   ##

fig,ax = plt.subplots(23,1) #
#fig,ax = plt.subplots(8,1) ##

for filename in glob.glob(directory + '\\'+ '*.wav'):
    with open(os.path.join(directory, filename), 'r') as f: 
        
        # Open the file
        riff = DICE.getRIFF(filename)
        fmt = DICE.getFMT(filename)
        data = DICE.getDATA(filename)
        
        # Extract the data
        y = data['samples']
        sampling_rate = data['SamplingRate']
        x = y[:, 0]
        x = x[0:800000]
        y1 = y[:, 1]
        y1 = y1[0:800000]
        #y2 = y[:,2]
        #y3 = y[:,3]
        
        #-----------------------------------------------------------------------------#
        '''
        # Feature-based NN: Feature extraction
        # Amplitude envelope, RMS energy and Spectral centroid
        list1 = []
        list2 = []
        freq_start = 1
        freq_stop = 100
        
        while freq_stop <= 3000:
            b, a = signal.butter(2, [freq_start,freq_stop],'bp',fs=sampling_rate, output='ba')
            filtered_signal = signal.lfilter(b, a, y1)
            rms_energy = rms(filtered_signal,frame_length=bit_size,hop_length=bit_size,center=False)
            rms_energy = np.resize(rms_energy,800)
            list1.append(rms_energy)
            amplitude_env = amplitude_envelope(filtered_signal,bit_size,bit_size)
            list2.append(amplitude_env)
            freq_start = freq_stop+1
            freq_stop = freq_stop+100
        rms_energy_dataset = (np.array(list1)).T
        amp_env_dataset = (np.array(list2)).T
        sc = spectral_centroid(y1, sr=sampling_rate, n_fft=bit_size, hop_length=bit_size, window=window, center=False)
        sc_dataset = np.reshape(sc,(800,1))
        feature_based_dataset = np.concatenate((amp_env_dataset,rms_energy_dataset,sc_dataset),axis=1)
        feature_based_dataset_comb.append(feature_based_dataset)
        
        #----------------------------------------------------------------------------#
        
        # Featureless NN
        # Time series dataset
        time_series = np.reshape(y1,(800,1000))
        time_series_dataset.append(time_series)
        
        # Spectrogram formation
        start = 0
        stop = bit_size
        mag_spectra, real_imag_spectra = ([] for i in range(2))
        while stop<=len(y1):
            win_frames = enframe(y1[start:stop], frame_size-frame_overlap, frame_size, window)
            stft_frames_x, mag_spectra_x, real_spectra_x, imag_spectra_x, phase_spectra_x = stft(win_frames, fft_size, sampling_rate)
            mag_spectra.append(mag_spectra_x)
            real_imag_x = np.dstack((real_spectra_x,imag_spectra_x))
            real_imag_spectra.append(real_imag_x)
            start = stop
            stop += bit_size
            
        mag_spectra = np.array(mag_spectra)
        real_imag_spectra = np.array(real_imag_spectra)
        
        mag_spectra_comb.append(mag_spectra)
        real_imag_spectra_comb.append(real_imag_spectra)
        '''
        #-----------------------------------------------------------------------------#
        
        # Label Extraction
        freq = center_freq[i]     #
        #freq = center_freq[0]    ##
        b, a = signal.butter(2, [freq-5,freq+5],'bp',fs=sampling_rate, output='ba')
        filtered_signal = signal.lfilter(b, a, y1)
        filtered_signal = preprocessing.scale(filtered_signal)
        start = 0
        end = bit_size
        label = []
        while end<=len(y1):
            if np.average(np.abs(filtered_signal[start:end]))>=0.6:
                dig = 1
            else:
                dig  = 0
            label.append(dig)
            start = end
            end += bit_size
        label = np.array(label)
        label_comb.append(label)
        
        target =np.repeat(label,bit_size)
        
# =============================================================================
#         if i<5:
#             plt.figure()
#             plt.plot(filtered_signal,alpha=0.5)
#             plt.plot(target)
# =============================================================================
        
        ax[i].plot(filtered_signal,alpha=0.7)
        if i==22:
            ax[i].set_xticks([0,200000,400000,600000,800000])
            ax[i].set_yticks([-4])
            ax[i].tick_params(axis='both', which='both', labelsize=20)
        elif i==0:
            ax[i].set_yticks([4])
            ax[i].set_xticks([])
            ax[i].tick_params(axis='y', which='both', labelsize=20)
        else:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        ax[i].tick_params(axis='y', which='both', labelsize=20)
        ax[i].set_ylim(-4,4)
        ax[i].set_ylabel('%i Hz' % center_freq[i],labelpad=50,fontsize=20,rotation=0)
        ax[i].yaxis.set_label_position("right")
        
# =============================================================================
#         ax[i].plot(filtered_signal,alpha=0.7)
#         if i==7:
#             ax[i].set_xticks([0,200000,400000,600000,800000])
#             ax[i].set_yticks([-2])
#             ax[i].tick_params(axis='both', which='both', labelsize=20)
#         elif i==0:
#             ax[i].set_yticks([2])
#             ax[i].set_xticks([])
#             ax[i].tick_params(axis='y', which='both', labelsize=20)
#         else:
#             ax[i].set_xticks([])
#             ax[i].set_yticks([])
#         ax[i].tick_params(axis='y', which='both', labelsize=20)
#         ax[i].set_ylim(-4,4)
#         ax[i].set_ylabel('%i mA' % amp[i],labelpad=50,fontsize=20,rotation=0)
#         ax[i].yaxis.set_label_position("right")
# =============================================================================
           
        #-----------------------------------------------------------------------------#
        
# =============================================================================
#         spectrogram = np.reshape(mag_spectra,(int(800*3),1024))
#         max_freq=int(sampling_rate/2)
#         dB_spectrogram = stft2level(spectrogram, int(max_freq*fft_size/sampling_rate))
#         max_time=x[-1]
#         plt.figure(figsize=(15, 8))
#         plt.imshow(np.transpose(np.array(dB_spectrogram)),origin='lower',extent=(0,max_time,0,max_freq),aspect='auto')
#         plt.xlabel('Time [s]')
#         plt.ylabel('Frequency [Hz]')
#         plt.title('Spectrogram')
# =============================================================================
        
        i = i+1
        print(i,filename)
        
        time +=x[-1]
   
fig.text(0.5, 0.04, 'Time-series samples', ha='center',fontsize=24)
fig.text(0.04, 0.5, 'Filtered signal', va='center', rotation='vertical',fontsize=24)
plt.show()

# =============================================================================
# # Combining and converting from list to numpy array
# df_time_series_dataset = np.array(time_series_dataset)
# df_time_series_dataset = np.reshape(df_time_series_dataset,((int(i*800)),1000))
# 
# df_feature_based_dataset_comb = np.array(feature_based_dataset_comb)
# df_feature_based_dataset_comb = np.reshape(df_feature_based_dataset_comb,((int(i*800)),61))
# 
# df_mag_spectra_comb = mag_spectra_comb[0]
# df_real_imag_spectra_comb = real_imag_spectra_comb[0]
# df_label_comb = label_comb[0]
# for i in range(len(label_comb)-1):
#     df_label_comb = np.append(df_label_comb,label_comb[i+1])
#     df_mag_spectra_comb = np.append(df_mag_spectra_comb,mag_spectra_comb[i+1],axis=0)
#     df_real_imag_spectra_comb = np.append(df_real_imag_spectra_comb,real_imag_spectra_comb[i+1],axis=0)
# =============================================================================
    
# =============================================================================
# np.save('Dataset7_amp-analysis_time_series_dataset_comb_1170Hz',df_time_series_dataset)
# np.save('amp-analysis_feature_based_dataset_comb_1170Hz',df_feature_based_dataset_comb)
# np.save('amp-analysis_mag_spectra_comb_1170Hz',df_mag_spectra_comb)
# np.save('amp-analysis_real_imag_spectra_comb_1170Hz',df_real_imag_spectra_comb)
# np.save('amp-analysis_label_comb_1170Hz',df_label_comb)
# =============================================================================

# =============================================================================
# np.save('Dataset2_freq-analysis_feature_based_dataset_comb_1A_comp',df_feature_based_dataset_comb)
# np.save('Dataset4_freq-analysis_mag_spectra_comb_1A_comp',df_mag_spectra_comb)
# np.save('Dataset6_freq-analysis_real_imag_spectra_comb_1A_comp',df_real_imag_spectra_comb)
# np.save('Dataset8_freq-analysis_time_series_dataset_comb_1A_comp',df_time_series_dataset)
# np.save('Label_freq-analysis_label_comb_1A_comp',df_label_comb)
# =============================================================================

# =============================================================================
# # Visualization
# spectrogram = np.reshape(df_mag_spectra_comb,(int(18400*3),1024))
# #spectrogram = df_mag_spectra_comb
# max_freq=int(sampling_rate/2)
# dB_spectrogram = stft2level(spectrogram, int(max_freq*fft_size/sampling_rate))
# max_time=int(time)
# plt.figure(figsize=(15, 8))
# plt.imshow(np.transpose(np.array(dB_spectrogram)),origin='lower',extent=(0,max_time,0,max_freq),aspect='auto')
# plt.xlabel('Time [s]',fontsize=24)
# plt.ylabel('Frequency [Hz]',fontsize=24)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# 
# =============================================================================
