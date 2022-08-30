import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
import torch
import pathlib

def preprocess(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    data = scaler.transform(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std


class dataLoader(Dataset):
    def __init__(self, data_name, mode,batch_size,seq_lenth, transform=None):
        self.data_name = data_name
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.seq_lenth = seq_lenth
        self.data = pd.read_csv(pathlib.Path('data',self.data_name,'labeled',self.mode,self.data_name+'.csv'))
        
        self.value = self.data.values[:,0:1]
        self.label = self.data.values[:,1:2]

        self.scaler = MinMaxScaler(feature_range=(0, 1)) 
        self.scaler.fit(self.value)
        self.value = self.scaler.transform(self.value)

        self.mean = self.value.mean(axis=0)
        self.std = self.value.std(axis=0)
        self.value = (self.value - self.mean)/self.std


        self.seq = []
        if self.mode == 'train'or self.mode == 'vali':
            for i in range(len(self.value)-seq_lenth):
                self.seq.append(self.value[i:i+seq_lenth,:])
        elif self.mode == 'test':
            self.value = self.value.reshape(len(self.value),1,1)
            
    def __len__(self):
        if self.mode == 'train'or self.mode == 'vali':
            return len(self.seq)-1
        elif self.mode == 'test':
            return len(self.value)-1
            
    def __getitem__(self, idx):
        if self.mode == 'train'or self.mode == 'vali':
            return self.seq[idx], self.seq[idx+1]
        elif self.mode == 'test':
            return self.value[idx], self.value[idx+1]
def get_dataloader(data_name, mode, batch_size,seq_lenth):
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    if mode == 'test':
        batch_size = 1
    dataset = dataLoader(data_name, mode,batch_size,seq_lenth)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=0)