from __future__ import division
import requests
import librosa
import librosa.display

import scipy.io.wavfile
import scipy.signal
from pydub import AudioSegment
import math
import wave
import soundfile as sf

from model import *
from dataset_wav import *
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import random
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import PIL

import matplotlib.pyplot as plt

rate = 8192
len_frame = 1024
len_hop = len_frame // 8

batch_size_train = 1
batch_size_val = 1
lr = 0.001
data_path = 'Data_test'
dataset = MIRDataset(data_path)
print("La taille du Dataset = ", len(dataset))
#train_set, val_set, set_notused = torch.utils.data.random_split(dataset, [3000, 540, 0]) # replace set_notused by test_set
#dataloader_train = DataLoader(train_set, batch_size_train, shuffle=True)
#dataloader_val = DataLoader(val_set, batch_size_val, shuffle=True)                 # non utilisé
device = torch.device("cpu")
print('------------', device)
loss_function = torch.nn.MSELoss().cuda()

load_model = True
# load model
if load_model:
  print("loading checkpoint")
  print('_________________')
  checkpoint = torch.load('unet_train_essai2.pth.tar')
  model = UNet()
  model.to(device)
  optimizer = optim.Adam(model.parameters(),lr)
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  model.to(device)

# prepare data
mix,vox = dataset[3]
mix = mix.to(device=device)                                             # tensor
vox = vox.to(device=device)
mix = torch.unsqueeze(mix, dim=0)


vox = torch.squeeze(vox)
print('vox squeeze', vox.shape)
vox = vox.detach().numpy()

# uncomment to display spectrogram

'''librosa.display.specshow(librosa.amplitude_to_db(vox, ref=np.max), x_axis='time')
plt.title('Power spectrogram after resize')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()'''

vox_rec = librosa.istft(vox, win_length=1022, hop_length=len_hop, center=True)
# save file voice
sf.write('test/test_one_voice.wav', vox_rec, rate, 'PCM_24')
# prediction
scores = model(mix)
mix = torch.squeeze(mix)
mix = mix.detach().numpy()
mix_init = librosa.istft(mix, win_length=1022, hop_length=len_hop, center=True)
# save file mixture
sf.write('test/test_one_mix.wav', mix_init, rate, 'PCM_24')

scores = torch.squeeze(scores)
scores = scores.detach().numpy()

# uncomment to display spectrogram of the reconstructed voice
'''librosa.display.specshow(librosa.amplitude_to_db(scores, ref=np.max), x_axis='time')
plt.title('Power spectrogram after resize')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()'''

scores_rec = librosa.istft(vox_rec, win_length=1022, hop_length=len_hop, center=True)
sf.write('test/test_one_reconstructedVoice.wav', scores_rec, rate, 'PCM_24')
