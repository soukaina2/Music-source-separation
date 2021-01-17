from __future__ import division
import torch
import torchaudio
import requests
import matplotlib.pyplot as plt

import numpy as np
import librosa
import librosa.display
from IPython.display import Audio

import scipy.io.wavfile
import scipy.signal
import os

from pydub import AudioSegment
import math
import wave


class SplitWavAudioMubin():
    def __init__(self, folder, folder_out, filename):
        self.folder = folder
        self.folder_out = folder_out
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]

        file_name = os.path.join(self.folder_out, self.filename.replace(".wav", split_filename))
        print("res ", file_name)
        split_audio.export(file_name + ".wav", format="wav")

    def multiple_split(self, sec_per_split):
        total_sec_init = math.floor(self.get_duration())
        print("Durée du fichier wav init = ", total_sec_init)
        total_sec = int(total_sec_init/sec_per_split) * sec_per_split
        print("Durée du fichier wav = ", total_sec)
        for i in range(0, total_sec, sec_per_split):
            n = int(i/sec_per_split)+1
            split_fn = '_' + str(n)
            self.single_split(i, i+sec_per_split, split_fn)
            print( ' Subfile ' + str(n) + ' Done')
            if i == total_sec - sec_per_split:
                print('All splited successfully')

## Get all the Wav files name (without folder)
def get_file_paths(dirname):
    file_paths = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            file_paths.append(filename)
    return file_paths



folder_in = 'data_init/vox'
sec_numb = 2
folder_out = "wavfile_2s/vox"
fname = get_file_paths(folder_in)
for i in fname:
  split_wav = SplitWavAudioMubin(folder_in, folder_out, i)
  split_wav.multiple_split(sec_per_split=sec_numb)
