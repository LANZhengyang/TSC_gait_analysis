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

import imp
LRP = imp.load_source('LRP', './LRP/__init__.py')
from LRP.lrp import LRP

import copy


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
    

class DL_classifier:
        def __init__(self,model):
            self.model = model
            
        def load(self,model_dir):
            self.model = self.model.load_from_checkpoint(model_dir)
            return self
            
        def fit(self, x_train, y_train, x_val, y_val, batch_size, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir=None):
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

                self.trainer = pl.Trainer(gpus=gpu,max_epochs= max_epochs, callbacks=[early_stop_callback], default_root_dir=default_root_dir)

            self.trainer.fit(self.model, data_loader_train, data_loader_test)

        def predict(self, x_pred, batch_size,gpu=[0]):
            pred_set = Dataset_torch(x_pred,with_label=False)
            data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=batch_size,num_workers=4)
            trainer = pl.Trainer(gpus=gpu)
            pred = trainer.predict(model=self.model,dataloaders = data_loader_pred)
            y_pre = torch.tensor([torch.argmax(i) for i in torch.cat(pred)])
            return y_pre
        
        def compute_LRP(self,x_test,y_test, rule='z_rule_no_bias'):
            
            model = self.model
            
            model = model.eval()
            
            lrp = LRP(model.double(), rule=rule)
            
            index_one = np.where(y_test==1)[0]
            index_zero = np.where(y_test==0)[0]

            pred_set_label_zero = Dataset_torch([x_test[index_zero],y_test[index_zero]])
            data_loader_pred_label_zero = torch.utils.data.DataLoader(dataset=pred_set_label_zero, batch_size=1,num_workers=4)
            
            relevance_list_label_zero = []
            for num, (image, label) in enumerate(data_loader_pred_label_zero):
                    relevance_list_label_zero.append(lrp(image).cpu().detach().numpy()[0])
                    
            relevance_list_label_zero = np.array(relevance_list_label_zero)
            r_mean_label_zero = np.mean(relevance_list_label_zero,axis=0)
            d_mean_label_zero = np.mean(x_test[index_zero],axis=0)
            
            max_mean_index_zero = np.max(r_mean_label_zero)
            min_mean_index_zero = np.min(r_mean_label_zero)
            
            
            pred_set_label_one = Dataset_torch((x_test[index_one],y_test[index_one]))
            data_loader_pred_label_one = torch.utils.data.DataLoader(dataset=pred_set_label_one, batch_size=1,num_workers=4)
            
            relevance_list_label_one = []
            for num, (image, label) in enumerate(data_loader_pred_label_one):
                    relevance_list_label_one.append(lrp(image).cpu().detach().numpy()[0])
                    
            relevance_list_label_one = np.array(relevance_list_label_one)
            r_mean_label_one = np.mean(relevance_list_label_one,axis=0)
            d_mean_label_one = np.mean(x_test[index_one],axis=0)
            
            max_mean_index_one = np.max(r_mean_label_one)
            min_mean_index_one = np.min(r_mean_label_one)
            
            lim_max = np.max([max_mean_index_one,max_mean_index_zero])
            lim_min = np.min([min_mean_index_one,min_mean_index_zero])
            
            return relevance_list_label_zero, relevance_list_label_one, r_mean_label_zero, r_mean_label_one, d_mean_label_zero, d_mean_label_one, lim_min, lim_max
          
            
    