import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LSTM(pl.LightningModule):
    def __init__(self,nb_classes, lr, lr_factor, lr_patience):
        super().__init__()
        
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        
        self.hidden_neurons = 400
        
        
        self.LSTM_1 = torch.nn.LSTM(input_size=22, hidden_size= self.hidden_neurons,bidirectional=True,batch_first=True)
        self.LSTM_2 = torch.nn.LSTM(input_size=800, hidden_size= self.hidden_neurons,bidirectional=True,batch_first=True)
        self.LSTM_3 = torch.nn.LSTM(input_size=800, hidden_size= self.hidden_neurons,bidirectional=True,batch_first=True)
        
        self.fc = nn.Linear(self.hidden_neurons * 2,3)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = torch.nn.Softmax()
        

    def forward(self,x):
        
        x = torch.transpose( x, 1, 2)        
       
        x, _ = self.LSTM_1(x)
        
        x = self.dropout(x)
        
        x, _ = self.LSTM_2(x)
        x = self.dropout(x)
        
        _, (x,_) = self.LSTM_3(x)
        
        x = torch.cat((x[-2,:,:], x[-1,:,:]), dim = 1)
        
        x = self.dropout(x)
        
        x = self.fc(x)  
        
        x = self.softmax(x)
        
        return x
         
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pre = self.forward(x.float())

        loss = nn.NLLLoss()(torch.log(y_pre),y_true.type(torch.cuda.LongTensor))
        
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch

        y_pre = self.forward(x.float())
            
        loss = nn.NLLLoss()(torch.log(y_pre),y_true.type(torch.cuda.LongTensor))

        self.log('test_loss', loss)

        return loss

    def predict_step(self, batch, batch_idx):
        
        x = batch

        y_pre = self.forward(x.float())
        
        return y_pre
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,eps=1e-07)
        scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, patience= self.lr_patience)

        return { 'optimizer': optimizer, 'lr_scheduler': { 'scheduler': scheduler, 'monitor': 'test_loss'} }

class Dataset_torch(Dataset):

    def __init__(self, data,with_label=True):
        self.with_label =  with_label
        
        if self.with_label:
            self.data_x, self.data_y = data
        else:
            self.data_x = data
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.with_label:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]
    
class Classifier_LSTM:
    def __init__(self, nb_classes, lr =0.001, lr_factor = 0.5, lr_patience=50):
        self.model = LSTM(nb_classes, lr, lr_factor, lr_patience)
        
    def fit(self, x_train, y_train, x_val, y_val, batch_size, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir = None):
        train_set = Dataset_torch([x_train, y_train])
        test_set = Dataset_torch([x_val, y_val])
        
        data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True,num_workers=4)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True,num_workers=4)
        
        
        if not earlystopping:
            self.trainer = pl.Trainer( gpus=gpu,max_epochs= max_epochs, default_root_dir= default_root_dir)
        
        elif earlystopping:
            early_stop_callback = EarlyStopping(
            monitor='test_loss',
            min_delta=0.00,
            patience=et_patience,
            verbose=True)
            
            self.trainer = pl.Trainer(  gpus=gpu,max_epochs= max_epochs, callbacks=[early_stop_callback], default_root_dir=default_root_dir)
        
        self.trainer.fit(self.model, data_loader_train, data_loader_test)
        
    def predict(self, x_pred, batch_size,gpu=[0]):
        pred_set = Dataset_torch(x_pred,with_label=False)
        data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=batch_size,num_workers=4)
        trainer = pl.Trainer(gpus=gpu)
        pred = self.trainer.predict(model=self.model,dataloaders = data_loader_pred)
        y_pre = torch.tensor([torch.argmax(i) for i in torch.cat(pred)])
        return y_pre
        