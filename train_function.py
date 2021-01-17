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


data_path = 'wavfile_2s'
batch_size_train = 8
batch_size_val = 8
lr = 0.0002
num_epochs = 100
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
dataloader_train = DataLoader(dataset, batch_size_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#loss_function = torch.nn.MSELoss().cuda()
loss_function = nn.L1Loss()


start = time.time()

model = UNet()
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(),lr)
losses = []
iters_train  = []
train_acc = []
n = 0 # the number of iterations

if load_model:
  print("loading checkpoint")
  checkpoint = torch.load('unet_train.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(num_epochs):
    print('---------------------training-----------------')
    '''saving checkpoints'''
    if epoch % 1 == 0:
        print('saving checkpoint')
        checkpoint = {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(checkpoint,'unet_train_essai2.pth.tar')
    running_loss = 0
    running_correct = 0
    for batch_idx,(mix,vox) in enumerate(dataloader_train):
        # get inputs and labels
        mix = mix.to(device=device)                                             # tensor
        vox = vox.to(device=device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        scores = model(mix)                                                     # tensor
        #loss + prediction
        loss = loss_function(scores,vox)
        loss.backward()
        optimizer.step()
        # save the current training information
        iters_train.append(n)
        losses.append(float(loss)/batch_size_train)  # compute *average* loss
        accuracy = get_accuracy(scores,vox)
        train_acc.append(accuracy) # compute training accuracy
        print('loss for batch',batch_idx,'  is ' ,loss/batch_size_train,' in epoch ',epoch)
        print('accuracy for batch',batch_idx,'  is ' ,accuracy,' in epoch ',epoch)
        # increment the iteration number
        n += 1

# print elapsed time
    time_elapsed = time.time() - start
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))

# plotting
plt.title("Training Curve (batch_size={}, lr={})".format(batch_size_train, lr))
plt.plot(iters_train, losses, label="Train")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig('fig/loss1.png')
#plt.show()
plt.close()

plt.title("Training Curve (batch_size={}, lr={})".format(batch_size_train, lr))
plt.plot(iters_train, train_acc, label="Train")
plt.xlabel("Iterations")
plt.ylabel("Training Accuracy")
plt.legend(loc='best')
plt.savefig('fig/train_acc1.png')
#plt.show()
plt.close()



print("Final Training Accuracy: {}".format(train_acc[-1]))
print('full error is:',np.sum(losses),'epoch %d' %epoch,' is finished')
print('full error is:',np.mean(losses),'epoch %d' %epoch,' is finished')
