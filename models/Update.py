#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import os

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def label_untargeted(label):
    #print(label)
    for i in range(len(label)):
        label[i] = ( label[i]+5 )%10
    #print(label)
    return label

# 换的时候把验证accuracy的也换了，最后算的时候要换回来。。。
def label_untargeted_back(label):
    for i in range(len(label)):
        label[i] = ( label[i]+5 )%10
    return label

def label_random(label):
    for i in range(len(label)):
        label[i] = random.randint(0,9)
    return label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, isUntargeted=False, isRandom=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.isUntargeted=isUntargeted
        self.isRandom = isRandom


    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #print(labels)
                #print(self.isUntargeted)
                if(self.isUntargeted):
                    labels = label_untargeted(labels)
                if (self.isRandom):
                    labels = label_random(labels)
                #print(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                #print(log_probs.tolist())
                predict_result = np.argmax(log_probs.tolist(), axis=1)
                #print(predict_result)
                #print(labels)
                accuracy = metrics.accuracy_score(labels.tolist(), predict_result)

                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                #batch_loss.append(loss.item())
                batch_loss.append(accuracy)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

