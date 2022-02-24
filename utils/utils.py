import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

from classifiers import ResNet
from classifiers import LSTM
from classifiers import InceptionTime
from classifiers import KNN_DTW


def load_dataset(dir_dataset,channel_first=True,flatten=False):
    index = [0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,29]
    
    npzfile = np.load(dir_dataset + 'train.npz')
    
    x_train_all = npzfile['Input']
    y_train = npzfile['Output']
    y_train = np.ravel(y_train)
    x_train = x_train_all[:,:,index]
    if channel_first:
        x_train = x_train.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'test.npz')
    x_test_all = npzfile['Input']
    y_test = npzfile['Output']
    y_test = np.ravel(y_test)
    x_test = x_test_all[:,:,index]
    if channel_first:
        x_test = x_test.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'pred.npz')
    x_pred_all = npzfile['Input']
    y_pred = npzfile['Output']
    y_pred = np.ravel(y_pred)
    x_pred = x_pred_all[:,:,index]
    if channel_first:
        x_pred = x_pred.transpose(0,2,1)
    
    if flatten == True:
        x_train = x_train.reshape([x_train.shape[0],-1])
        x_test = x_test.reshape([x_test.shape[0],-1])
        x_pred = x_pred.reshape([x_pred.shape[0],-1])
    
    nb_classes = len(np.unique(np.concatenate((y_train, y_test, y_pred), axis=0)))
    
    return x_train, y_train ,x_test, y_test, x_pred, y_pred, nb_classes

class Dataset_torch(Dataset):

    def __init__(self, data):
        self.data_x, self.data_y = data
    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

def gait_dataloader(data_type,dataset,root_path=r'archives/AQM/',batch_size=64,flatten=False):
    data_all = load_dataset(root_path+dataset+'/',flatten=flatten)
    nb_classes = data_all[6]
    
    if data_type == 'numpy':
        return data_all
    elif data_type == 'torch':
        train_set = Dataset_torch(data_all[0:2])
        test_set = Dataset_torch(data_all[2:4])
        pred_set = Dataset_torch(data_all[4:6])

        data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True,num_workers=4)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True,num_workers=4)
        data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=batch_size,num_workers=4)
        
        return data_loader_train, data_loader_test, data_loader_pred

    
    
def init_model(model_name=None,model=None,nb_classes=2 ,lr =0.001, lr_factor = 0.5, lr_patience=50):
    if model_name == None:
        return model
    elif model_name != None:
        if model_name == 'ResNet':
            return ResNet.Classifier_ResNet(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience)
        elif model_name == 'LSTM':
            return LSTM.Classifier_LSTM(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience)
        elif model_name == 'InceptionTime':
            return InceptionTime.Classifier_InceptionTime(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience)
        elif model_name == 'KNN_DTW':
            return KNN_DTW.Classifier_KNN_DTW()
        
