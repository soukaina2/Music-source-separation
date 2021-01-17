import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import requests
import librosa
import scipy.io.wavfile
import scipy.signal
from pydub import AudioSegment
import math
import wave

'''Data class'''
class MIRDataset(Dataset):
    """ Stereo Dataset. """
    def __init__(self, data_path, transform=None):
        self.data_vox = glob.glob(os.path.join(data_path, 'vox', '*.wav'))
        self.transform = transform

    def __len__(self):
        return len(self.data_vox)



    def __getitem__(self, idx, rate = 8192, len_frame = 1024):
        ###############
        #### VOIX #####
        ###############
        #read wav
        path_vox = self.data_vox[idx]
        #print('vox path', path_vox)
        ## compute spectrogram
        vox_wav, sample_rate = np.array(librosa.load(path_vox, rate))
        vox_spec = librosa.stft(vox_wav, n_fft=len_frame, hop_length=len_frame//8)
        ## change type
        vox_spec = vox_spec.astype(np.float32)
        #vox_spec = vox_spec/255
        vox_spec = np.delete(vox_spec,np.s_[512:],axis=0)
        vox_spec = np.delete(vox_spec,np.s_[128:],axis=1)
        # conversion numpy -> tensor
        vox_spec = torch.from_numpy(vox_spec)
        vox_spec = torch.unsqueeze(vox_spec, dim=0)

        ##########
        ### MIX ##
        ##########
        # read mix image
        path_mix = path_vox.replace('vox', 'mix')
        path_mix = path_mix[:len(path_mix)-9]+'Mix'+path_mix[len(path_mix)-6:]
        #path_mix = path_vox.replace('Vox', 'Mix')
        #print('mix path', path_mix)
        ## compute spectrogram
        mix_wav, sample_rate = np.array(librosa.load(path_mix, rate))
        mix_spec = librosa.stft(mix_wav, n_fft=len_frame, hop_length=len_frame//8)
        ## change type
        mix_spec = mix_spec.astype(np.float32)
        #mix_spec = mix_spec/255
        mix_spec = np.delete(mix_spec,np.s_[512:],axis=0)
        mix_spec = np.delete(mix_spec,np.s_[128:],axis=1)
        # conversion numpy -> tensor
        mix_spec = torch.from_numpy(mix_spec)
        mix_spec = torch.unsqueeze(mix_spec, dim=0)
        

        return mix_spec, vox_spec
