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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

import multiprocessing
from sklearn.model_selection import KFold

def load_dataset(dir_dataset,channel_first=True,flatten=False,shuffle=False,random_state=0):
    index = [0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,29]
    
    npzfile = np.load(dir_dataset + 'train.npz')
    
    x_train_all = npzfile['Input']
    y_train = npzfile['Output']
    y_train = np.ravel(y_train)
    x_train = x_train_all[:,:,index]
    if channel_first:
        x_train = x_train.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'test.npz')
    x_val_all = npzfile['Input']
    y_val = npzfile['Output']
    y_val = np.ravel(y_val)
    x_val = x_val_all[:,:,index]
    if channel_first:
        x_val = x_val.transpose(0,2,1)
    
    npzfile = np.load(dir_dataset + 'pred.npz')
    x_test_all = npzfile['Input']
    y_test = npzfile['Output']
    y_test = np.ravel(y_test)
    x_test = x_test_all[:,:,index]
    if channel_first:
        x_test = x_test.transpose(0,2,1)
    
    if flatten == True:
        x_train = x_train.reshape([x_train.shape[0],-1])
        x_val = x_val.reshape([x_val.shape[0],-1])
        x_test = x_test.reshape([x_test.shape[0],-1])
    
    nb_classes = len(np.unique(np.concatenate((y_train, y_val, y_test), axis=0)))
    
    if shuffle == True:
        x_data = np.concatenate([x_train,x_val,x_test])
        y_data = np.concatenate([y_train,y_val,y_test])
        
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=random_state)
        x_test, x_val, y_test, y_val = train_test_split(x_data, y_data, test_size=0.25, random_state=random_state)
    
    return x_train, y_train ,x_val, y_val, x_test, y_test, nb_classes

class Dataset_torch(Dataset):

    def __init__(self, data):
        self.data_x, self.data_y = data
    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

def gait_dataloader(data_type,dataset,root_path=r'archives/AQM/',batch_size=64,flatten=False,shuffle=False,random_state=0):
    data_all = load_dataset(root_path+dataset+'/',flatten=flatten,shuffle=shuffle,random_state=random_state)
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

def accuracy_end_to_end(model,x_train,y_train,x_test,y_test,x_val=None,y_val=None,batch_size=None,max_epochs=None):
    if 'sklearn' in str(type(model)) or 'sktime' in str(type(model)): 
        return accuracy_score(y_test,model.fit(x_train,y_train).predict(x_test))
    else:
        return accuracy_score(y_test ,model.fit(x_train,y_train,batch_size,x_val,y_val,max_epochs=max_epochs).predict(x_test,batch_size=64))
    
    
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
        
class GridSearchCV:
    def __init__(self, model,para_list,n_splits=7,n_process=10):
        self.model = model
        self.para_list = list(ParameterGrid(para_list))
        self.n_splits = n_splits
        self.n_process = n_process
        
    def fit(self, x_data_cv, y_data_cv):
        pool = multiprocessing.Pool(processes=self.n_process)
    
        kf = KFold(n_splits=self.n_splits)
        
        self.accuracy_mean_list = []
        self.accuracy_std_list = []
        
        self.accuracy_list = []
        
        if 'sklearn' in str(type(self.model)) or 'sktime' in str(type(self.model)): 
            for para in self.para_list:

                accuracy = []
                for train_index, test_index in kf.split(x_data_cv):


                    accuracy.append( pool.apply_async(accuracy_end_to_end,args=(self.model(**para),x_data_cv[train_index], y_data_cv[train_index],x_data_cv[test_index],y_data_cv[test_index])))
    #                 accuracy.append( accuracy_end_to_end(self.model(**para),x_data_cv[train_index], y_data_cv[train_index],x_data_cv[test_index],y_data_cv[test_index]))
                accuracy = [i.get() for i in accuracy]

                self.accuracy_list.append(accuracy)

                self.accuracy_mean_list.append(np.mean(accuracy))
                self.accuracy_std_list.append(np.std(accuracy))

                print('parameter:',para)
                print('accuracy:',np.mean(accuracy),"±",np.std(accuracy))
                print()

            pool.close() 
            pool.join()
        else:
            for para in self.para_list: 

                accuracy = []
                for train_index, test_index in kf.split(x_data_cv):


                    accuracy.append( accuracy_end_to_end(self.model(**para),x_data_cv[train_index], y_data_cv[train_index],x_data_cv[test_index],y_data_cv[test_index]))

                self.accuracy_list.append(accuracy)

                self.accuracy_mean_list.append(np.mean(accuracy))
                self.accuracy_std_list.append(np.std(accuracy))

                print('parameter:',para)
                print('accuracy:',np.mean(accuracy),"±",np.std(accuracy))
                print()
            
        
        self.best_accuracy_mean = self.accuracy_mean_list[np.argmax(self.accuracy_mean_list)]
        self.best_accuracy_std = self.accuracy_std_list[np.argmax(self.accuracy_mean_list)]
        self.best_para = self.para_list[np.argmax(self.accuracy_mean_list)]
        print('------------------------')
        print('best parameter:',self.best_para)
        print('accuracy:',self.best_accuracy_mean,"±",self.best_accuracy_std)
        
        self.accuracy_list = np.array(self.accuracy_list)
        