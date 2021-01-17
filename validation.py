from model import *
from dataset_wav import *
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
import numpy as np

import os
import glob
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import PIL

import matplotlib.pyplot as plt

import time


data_path = 'Data_test'
batch_size_val = 8
lr = 0.001
num_epochs = 5
load_model = False
###

def get_accuracy(scores, vox):
    """
    Return the accuracy of the model on the input data and actual ground truth.
    """
    sigmoid = nn.Sigmoid()
    scores = sigmoid(scores)
    vox = sigmoid(vox)
    pred = (scores > 0.5).type(torch.FloatTensor)
    actual = (vox > 0.5).type(torch.FloatTensor)
    correct = (pred == actual).type(torch.FloatTensor) #pred == actual returns false or true for each element of the tensor
    return float(torch.mean(correct))

dataset = MIRDataset(data_path)
print("La taille du Dataset = ", len(dataset))
dataloader_val = DataLoader(dataset, batch_size_val)
device = torch.device("cpu")
print(device)
#loss_function = torch.nn.MSELoss().cuda()
loss_function = nn.L1Loss()


start = time.time()

model = UNet()
model.to(device)
model.eval()
optimizer = optim.Adam(model.parameters(),lr)
losses = []
iters_val  = []
val_acc = []
n = 0 # the number of iterations


for epoch in range(num_epochs):
    print('---------------------Validation-----------------')
    '''loading checkpoints'''
    checkpoint = torch.load('unet_train_essai2.pth.tar')
    model = UNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for batch_idx,(mix,vox) in enumerate(dataloader_val):
        # get inputs and labels
        mix = mix.to(device=device)                                             # tensor
        vox = vox.to(device=device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        scores = model(mix)                                                     # tensor
        # save the current training information
        iters_val.append(n)
        accuracy = get_accuracy(scores,vox)
        val_acc.append(accuracy) # compute training accuracy
        print('accuracy for batch',batch_idx,'  is ' ,accuracy,' in epoch ',epoch)
        # increment the iteration number
        n += 1

# print elapsed time
time_elapsed = time.time() - start
print("Training complete in {:.0f}m {:.0f}s".format(
    time_elapsed // 60, time_elapsed % 60))

# plotting

plt.title("Validation Curve (batch_size={}, lr={})".format(batch_size_val, lr))
plt.plot(iters_val, val_acc, label="validation")
plt.xlabel("Iterations")
plt.ylabel("Validation Accuracy")
plt.legend(loc='best')
plt.show()

print("Final Training Accuracy: {}".format(val_acc[-1]))
print('full error is:',np.sum(losses),'epoch %d' %epoch,' is finished')
print('full error is:',np.mean(losses),'epoch %d' %epoch,' is finished')
