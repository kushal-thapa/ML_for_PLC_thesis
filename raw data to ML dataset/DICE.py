# WAV file manipulation from Matlab equivalent
# SAM 07/05/19 - Initial version

## mmap discussion is interesting...
## https://stackoverflow.com/questions/1035340/reading-binary-file-and-looping-over-each-byte#

import numpy as np

# Opens file, reads RIFF chunk from WAV file header
# Input:
#   filename.wav ... a WAV file
# Reads:
#   RIFF header from file

def getRIFF(filename):
    
    riff = {}
    riff['error'] = 0
    
    with open(filename,'rb') as f:
        riff['tag'] = f.read(4).decode('utf8')
        if (riff['tag'] != 'RIFF'):
            riff['error'] = -1
            return(riff)
        
        dum = f.read(4)
        riff['len'] = int.from_bytes(dum, byteorder='little', signed=False)

        riff['wav'] = f.read(4).decode('utf8')
        if (riff['wav'] != 'WAVE'):
            riff['error'] = -1
            return(riff)

    return(riff)

# Opens file, reads FMT chunk from WAV file header
# Input:
#   filename.wav ... a WAV file
# Reads:
#   FMT header from file

def getFMT(filename):

    fmt = {}
    fmt['error'] = 0
    
    with open(filename,'rb') as f:
        # Skip RIFF header .. 12 bytes @ front of file
        f.seek(12,0);
        
        # must be 'fmt ' (char)
        fmt['tag'] = f.read(4).decode('utf8')
        if (fmt['tag'] != 'fmt '):
            fmt['error'] = -1
            return(fmt)
        
        # number of bytes in the following FMT chunk (int)
        # len should be 16 for PCM data.
        dum = f.read(4)
        fmt['len'] = int.from_bytes(dum, byteorder='little', signed=False)
        
        # PCM data has "AudioFormat" = 1, other values indicate compression
        # DICE seems to put a "-2" here ... not sure what that means (uint16)
        dum = f.read(2)
        fmt['AudioFormat'] = int.from_bytes(dum, byteorder='little', signed=True)
        
        # channels .. Mono=1, Stereo=2, etc (uint16)
        dum = f.read(2)
        fmt['NumChannels'] = int.from_bytes(dum, byteorder='little', signed=False)
        
        # Sampling Rate in samples/second (int32)
        dum = f.read(4)
        fmt['SamplesPerSec'] = int.from_bytes(dum, byteorder='little', signed=False)
        
        # AvgBytesPerSec = SamplesPerSec * NumChannels * BitsPerSample/8 (int32)
        dum = f.read(4)
        fmt['AvgBytesPerSec'] = int.from_bytes(dum, byteorder='little', signed=False)
        
        # BlockAlign = NumChannels * BitsPerSample/8 (int16)
        # Total number of bytes for one sample, incl. all channels
        dum = f.read(2)
        fmt['BlockAlign'] = int.from_bytes(dum, byteorder='little', signed=False)
        
        # Bits per sample, single-channel (int16)
        dum = f.read(2)
        fmt['BitsPerSample'] = int.from_bytes(dum, byteorder='little', signed=False)

        # Read WAVE-EXTENDED header stuff, if present
        # not sure what the format of the extra WAV-EXT header stuff is 
        # for DICE ... apparently NI throws something else in the mix

        # "fact" in single-channel DICE files ???
        if (fmt['AudioFormat'] == 1):
            dum = f.read(4).decode('utf8')
            if (dum == 'fact'):
                fmt['ExtraStuff'] = f.read(8)
                fmt['ExtraLen'] = 12
            else:
                fmt['ExtraLen'] = 0
            
        if (fmt['AudioFormat'] != 1):
            dum = f.read(2)
            fmt['ExtraLen'] = int.from_bytes(dum, byteorder='little', signed=False)
            fmt['ExtraStuff'] = f.read(fmt['ExtraLen']) # uchar

    return(fmt)
        
# Opens file, reads DATA chunk from WAV file 
# Input:
#   filename.wav ... a WAV file
#   offset ... bytes from beginning of file to start of DATA header
# Reads:
#   DATA header from file

def getDATA(filename):

    data = {}
    fmt = getFMT(filename)
    if (fmt['error'] < 0): return(fmt)
            
    with open(filename,'rb') as f:
        
        # Skip RIFF & FMT headers
        if (fmt['AudioFormat'] == 1):
            # This is the PCM format (fmt.len = 16)
            offset = 12 + 8 + fmt['len'] + fmt['ExtraLen']
        elif (fmt['AudioFormat'] == -2):
            # This is the DICE format (fmt.len = 40)
            # Not sure why NI has the extra 22 bytes
            offset = 10 + fmt['len'] + fmt['ExtraLen']
        else:
            data['error'] = -1;
            return(data)

        # Skipping ...
        data['offset'] = offset
        f.seek(offset,0)
        
        # Read data header (4 bytes, uchar) .. must be "data"      
        data['tag'] = f.read(4).decode('utf8')
        if (data['tag'] != 'data'):
            data['error'] = -1
            return(data)
        
        # Read data length (4 bytes, uint)
        dum = f.read(4)
        data['len'] = int.from_bytes(dum, byteorder='little', signed=False)
        
        # need to check data type with BitsPerSample
        # here, assume always int16 

        SamplesPerChan = int(data['len'] / fmt['BlockAlign'])
        NumChannels = fmt['NumChannels']
        TotalSamples = int(SamplesPerChan * NumChannels)
                           
        data['SamplesPerChan'] = SamplesPerChan
        data['NumChannels'] = NumChannels
        data['TotalSamples'] = TotalSamples
        data['SampleTime'] = 1.0/fmt['SamplesPerSec']
        data['SamplingRate'] = fmt['SamplesPerSec']

        s = np.fromfile(f,dtype='int16')
        ss = s.reshape(SamplesPerChan,NumChannels)

        secs = SamplesPerChan / fmt['SamplesPerSec']
        t = np.linspace(0,secs,SamplesPerChan)
        tt = t.reshape(SamplesPerChan,1)

        samples = np.hstack([tt,ss])

        data['samples'] = samples
        
    return(data)
