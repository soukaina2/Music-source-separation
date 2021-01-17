from __future__ import division
import torch
#import torchaudio
#torchaudio.set_audio_backend("sox_io")
import requests
import matplotlib.pyplot as plt

import numpy as np
import librosa
import librosa.display
from IPython.display import display
from IPython.display import Audio

import scipy.io.wavfile
import scipy.signal
import os

from pydub import AudioSegment
import math
import wave
import matplotlib.pyplot as plt

import numpy as np
import soundfile as sf


## Parameters
path_vox = "wavfile_2s/mix/abjones_1_01Mix_2.wav"
rate = 8192
len_frame = 1024
len_hop = len_frame // 8

## load and play wav file
vox_wav, sample_rate = np.array(librosa.load(path_vox, rate))
#display(Audio(vox_wav, rate=rate))
## informations
# print("sample rate =", sample_rate)
# print("size of vox_wav =", len(vox_wav))
# print("type of vox_wav is :", vox_wav.dtype)
# print("shape of vox_wav :", vox_wav.shape)
## compute spectrogram
vox_spec = librosa.stft(vox_wav, n_fft=len_frame, hop_length=len_hop)
## change type
vox_spec = vox_spec.astype(np.float32)

## show
librosa.display.specshow(librosa.amplitude_to_db(vox_spec, ref=np.max), x_axis='time')
plt.title('Power spectrogram before resize')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
print("vox_spec shape :", vox_spec.shape)
## resize
vox_spec = np.delete(vox_spec,np.s_[512:],axis=0)
vox_spec = np.delete(vox_spec,np.s_[144:],axis=1)
print("vox_spec new shape :", vox_spec.shape)
## show
#plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(vox_spec, ref=np.max), x_axis='time')
plt.title('Power spectrogram after resize')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
## convert to wav
vox_rec = librosa.istft(vox_spec, win_length=1022, hop_length=len_hop, center=True)
#librosa.output.write_wav('test/test_reconstructes.wav', vox_rec, rate)

sf.write('test/test_reconstructes.wav', vox_rec, rate, 'PCM_24')
#display(Audvox_recio(vox_rec, rate=rate))
