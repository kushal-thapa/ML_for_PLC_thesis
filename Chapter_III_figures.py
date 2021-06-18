# Importing libraries
from scipy import signal
import DICE
import matplotlib.pyplot as plt
import numpy as np
#import scipy.fftpack as fftpack 
import cmath
from scipy.fftpack import fft, fftfreq
from sklearn import preprocessing

plt.style.use('classic')

#################################################################################
# Open the file
filename = '1595Hz.wav'

riff = DICE.getRIFF(filename)
fmt = DICE.getFMT(filename)
data = DICE.getDATA(filename)

# Get data
PhaseA = 1
PhaseB = 2
PhaseC = 3
y = data['samples']
x = y[:, 0]
fs = data['SamplingRate']
Phase_A_data = y[:,PhaseA]
Phase_B_data = y[:,PhaseB]
Phase_C_data = y[:,PhaseC]
Ts = 1/fs

#################################################################################

# Label Extraction
freq = 1595
b, a = signal.butter(2, [freq-5,freq+5],'bp',fs=fs, output='ba')
filtered_signal = signal.lfilter(b, a,Phase_A_data)
filtered_signal = preprocessing.scale(filtered_signal)
start = 0
end = 1000
label = []
while end<=len(Phase_A_data):
    if np.average(np.abs(filtered_signal[start:end]))>=0.6:
        dig = 1
    else:
        dig  = 0
    label.append(dig)
    start = end
    end += 1000
label = np.array(label)

target =np.repeat(label,1000)

plt.figure()
plt.plot(filtered_signal,alpha=0.7,label='Filtered signal')
plt.plot(target,'r',linewidth=2,label='Extracted label')
plt.xlabel('Samples',fontsize=24)
plt.ylabel('Magnitude',fontsize=24)
plt.xlim(0,800000)
plt.ylim(-4,4)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

#################################################################################
'''
# Time domain plot
plt.figure(1)
plt.plot(x[0:500], Phase_A_data[0:500], 'b-', linewidth=4, label='Phase A')
plt.plot(x[0:500], Phase_B_data[0:500], 'r--', linewidth=4,label='Phase B')
plt.plot(x[0:500], Phase_C_data[0:500], 'g:', linewidth=4, label='Phase C')

plt.xlabel('Time [s]',fontsize=24)
plt.ylabel('Amplitude [mV]',fontsize=24)
#plt.title('DICE data',fontsize=14)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

#################################################################################

# Frequency-domain analysis

# defining a function that performs fft
def magnitude_response(numerator,Fs):
    nfft=int(8*(2**(np.ceil(np.log2(len(numerator))))))
    num_fft=fft(numerator,nfft)
    energy=np.dot(abs(num_fft),abs(num_fft))/len(num_fft)
    energy=float("{0:.2f}".format(energy))
    fft_freq=fftfreq(nfft,1/Fs)
    fft_freq=fft_freq[0:nfft//2]
    num_fft=10*np.log10(abs(2*num_fft[0:nfft//2]))
    return fft_freq, num_fft, energy

fft_freq,num_fft,energy = magnitude_response(Phase_A_data,fs)
plt.figure()
plt.plot(fft_freq,num_fft)
plt.xlabel('Frequency [Hz]',fontsize=24)
plt.ylabel('Magnitude [dB]',fontsize=24)
plt.xticks((range(0,2001,200)),fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,2000)
plt.grid()
plt.show()

#################################################################################

# Spectrograms

# Functions for spectrogram generation
def enframe(x,S,L,wdw):
    # Divides the time series signal 'x' into multiple frames of length 'L' 
    # with frame overlap (L-S) and applies window 'wdw'
    # Outputs the frames
    w = signal.get_window(wdw,L)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    for t in range(0,nframes):
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
    return(frames)
def stft(frames,N,Fs):
    # Does short term fourier transform of length 'N' to each frame of 'frames' 
    # containing time-series data of sampling frequency 'Fs'
    # Outputs the magnitude, real, imaginary and phase spectra of each frame
    stft_frames = [ fft(x,N) for x in frames]
    freq_axis = np.linspace(0,Fs,N)
    mag_spectra = [ abs(x) for x in stft_frames ]
    real_spectra = [ x.real for x in stft_frames ]
    imag_spectra = [ x.imag for x in stft_frames ]
    phase_spectra = []
    for i in range(0,len(stft_frames)):
        phase_spectra.append([ cmath.phase(x) for x in stft_frames[i] ])
    return(mag_spectra, real_spectra, imag_spectra, phase_spectra, freq_axis)
def stft2level(stft_spectra,max_freq_bin):
    # Converts the spectral values to decibels
    max_magnitude = max([ max(x) for x in stft_spectra ])
    min_magnitude = max_magnitude / 5000.0
    for t in range(0,len(stft_spectra)):
        for k in range(0,len(stft_spectra[t])):
            stft_spectra[t][k] /= min_magnitude
            if stft_spectra[t][k] < 1:
                stft_spectra[t][k] = 1
    level_spectra = [ 10*np.log10(x[0:max_freq_bin]) for x in stft_spectra ]
    return(level_spectra)
def sgram(x,window,frame_skip,frame_length,fft_length, fs, max_freq):
    # Combines the above definitions to output magnitude, real, imaginary and phase spectrograms
    frames = enframe(x,frame_skip,frame_length,window)
    (mag_spectra, real_spectra, imag_spectra, phase_spectra, freq_axis) = stft(frames, fft_length, fs)
    sgram_mag = stft2level(mag_spectra, int(max_freq*fft_length/fs))
    sgram_real = stft2level(real_spectra, int(max_freq*fft_length/fs))
    sgram_imag = stft2level(imag_spectra, int(max_freq*fft_length/fs))
    sgram_phase = stft2level(phase_spectra, int(max_freq*fft_length/fs))    
    max_time = len(frames)*frame_skip/fs
    return(sgram_mag, sgram_real, sgram_imag, sgram_phase, max_time, max_freq)

# =============================================================================

# Generating spectrograms
sgram_mag_A, sgram_real_A, sgram_imag_A, sgram_phase_A, max_time, max_freq = sgram(Phase_A_data,'hanning',750,1000,1024,fs,4000)

# Plotting
plt.figure(1,figsize=(15, 8))
plt.imshow(np.transpose(np.array(sgram_mag_A)),origin='lower',extent=(0,max_time,0,max_freq),aspect='auto')
plt.xlabel('Time [s]',fontsize=24)
plt.ylabel('Frequency [Hz]',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Constructed Spectrogram: Magnitude',fontsize=26)

plt.figure(2,figsize=(15, 8))
plt.imshow(np.transpose(np.array(sgram_real_A)),origin='lower',extent=(0,max_time,0,max_freq),aspect='auto')
plt.xlabel('Time [s]',fontsize=24)
plt.ylabel('Frequency [Hz]',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Constructed Spectrogram: Real')
plt.figure(3,figsize=(15, 8))
plt.imshow(np.transpose(np.array(sgram_imag_A)),origin='lower',extent=(0,max_time,0,max_freq),aspect='auto')
plt.xlabel('Time [s]',fontsize=24)
plt.ylabel('Frequency [Hz]',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Constructed Spectrogram: Imaginary')
plt.figure(4,figsize=(15,8))
plt.imshow(np.transpose(np.array(sgram_phase_A)),origin='lower',extent=(0,max_time,0,max_freq),aspect='auto')
plt.xlabel('Time [s]',fontsize=24)
plt.ylabel('Frequency [Hz]',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Constructed Spectrogram: Phase')
'''
