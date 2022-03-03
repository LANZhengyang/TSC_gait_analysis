import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

from classifiers.DL_classifier import DL_classifier

class ResNet(pl.LightningModule):
    def __init__(self,nb_classes, lr, lr_factor, lr_patience):
        super().__init__()
        
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        
        # Block 1
        self.conv1 = nn.Conv1d(22, 22, 8)
        self.conv2 = nn.Conv1d(22, 22, 5)
        self.conv3 = nn.Conv1d(22, 22, 3)
        self.conv4 = nn.Conv1d(22, 22, 1)
        self.BN1 = nn.BatchNorm1d(22)
        self.BN2 = nn.BatchNorm1d(22)
        self.BN3 = nn.BatchNorm1d(22)
        self.BN4 = nn.BatchNorm1d(22)
        self.relu = nn.ReLU()
        
        # Block 2
        
        self.conv5 = nn.Conv1d(22, 22, 8)
        self.conv6 = nn.Conv1d(22, 22, 5)
        self.conv7 = nn.Conv1d(22, 22, 3)
        self.conv8 = nn.Conv1d(22, 22, 1)
        self.BN5 = nn.BatchNorm1d(22)
        self.BN6 = nn.BatchNorm1d(22)
        self.BN7 = nn.BatchNorm1d(22)
        self.BN8 = nn.BatchNorm1d(22)
        
        # Block 3
        
        self.conv9 = nn.Conv1d(22, 22, 8)
        self.conv10 = nn.Conv1d(22, 22, 5)
        self.conv11 = nn.Conv1d(22, 22, 3)
        self.conv12 = nn.Conv1d(22, 22, 1)
        self.BN9= nn.BatchNorm1d(22)
        self.BN10 = nn.BatchNorm1d(22)
        self.BN11 = nn.BatchNorm1d(22)
        self.BN12 = nn.BatchNorm1d(22)
        
        # Final
        
        self.globalavgpooling1d = nn.AdaptiveAvgPool1d(1)
        self.fc_output = nn.Linear(22, nb_classes)
        self.softmax = torch.nn.Softmax()
        
        # test save
        self.accuracy_test = 0
        

    def forward(self,x):
        input_layer = x
        
        # Block 1
        
        x = F.pad(input_layer, (3,4,0,0)) # [left, right, top, bot]
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        
        y = F.pad(x, (2,2,0,0)) # [left, right, top, bot]
        y = self.conv2(y)
        y = self.BN2(y)
        y = self.relu(y)
        
        z = F.pad(y, (1,1,0,0)) # [left, right, top, bot]
        z = self.conv3(z)
        z = self.BN3(z)
        
        shortcut_y = self.conv4(input_layer)
        shortcut_y = self.BN4(z)
        
        output_block_1 = torch.add(shortcut_y,z)
        output_block_1 = self.relu(output_block_1)
        
        # Block 2
        
        x = F.pad(output_block_1, (3,4, 0, 0)) # [left, right, top, bot]
        x = self.conv5(x)
        x = self.BN5(x)
        x = self.relu(x)
        
        y = F.pad(x, (2, 2 , 0, 0 )) # [left, right, top, bot]
        y = self.conv6(y)
        y = self.BN6(y)
        y = self.relu(y)
        
        z = F.pad(y, ( 1, 1 , 0, 0)) # [left, right, top, bot]
        z = self.conv7(z)
        z = self.BN7(z)
        
        shortcut_y = self.conv8(output_block_1)
        shortcut_y = self.BN8(shortcut_y)
        
        output_block_2 = torch.add(shortcut_y,z)
        output_block_2 = self.relu(output_block_2)
        
        # Block 3
        
        x = F.pad(output_block_2, (3,4,0,0)) # [left, right, top, bot]
        x = self.conv9(x)
        x = self.BN9(x)
        x = self.relu(x)
        
        y = F.pad(x, (2,2,0,0)) # [left, right, top, bot]
        y = self.conv10(y)
        y = self.BN10(y)
        y = self.relu(y)
        
        z = F.pad(y, (1,1,0,0)) # [left, right, top, bot]
        z = self.conv11(z)
        z = self.BN7(z)
        
        shortcut_y = self.BN8(output_block_2)
        
        output_block_3 = torch.add(shortcut_y,z)
        output_block_3 = self.relu(output_block_3)
        
        # Final
        
        gap_layer = self.globalavgpooling1d(output_block_3)
        gap_layer = torch.squeeze(gap_layer)
        output_layer = self.fc_output(gap_layer).float()
        output_layer = self.softmax(output_layer)
        
        return output_layer.float()
         
    
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

# class Dataset_torch(Dataset):

#     def __init__(self, data,with_label=True):
#         self.with_label =  with_label
        
#         if self.with_label:
#             self.data_x, self.data_y = data
#         else:
#             self.data_x = data
#     def __len__(self):
#         return len(self.data_x)

#     def __getitem__(self, idx):
#         if self.with_label:
#             return self.data_x[idx], self.data_y[idx]
#         else:
#             return self.data_x[idx]
    
class Classifier_ResNet(DL_classifier):
    def __init__(self, nb_classes, lr =0.001, lr_factor = 0.5, lr_patience=50):
        self.model = ResNet(nb_classes, lr, lr_factor, lr_patience)
        super().__init__(self.model)
        
#     def fit(self, x_train, y_train, x_val, y_val, batch_size, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir = None):
#         train_set = Dataset_torch([x_train, y_train])
#         test_set = Dataset_torch([x_val, y_val])
        
#         data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True,num_workers=4)
#         data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True,num_workers=4)
        
        
#         if not earlystopping:
#             self.trainer = pl.Trainer( gpus=gpu,max_epochs= max_epochs, default_root_dir= default_root_dir)
        
#         elif earlystopping:
#             early_stop_callback = EarlyStopping(
#             monitor='test_loss',
#             min_delta=0.00,
#             patience=et_patience,
#             verbose=True)
            
#             self.trainer = pl.Trainer(  gpus=gpu,max_epochs= max_epochs, callbacks=[early_stop_callback], default_root_dir=default_root_dir)
        
#         self.trainer.fit(self.model, data_loader_train, data_loader_test)
        
#     def predict(self, x_pred, batch_size,gpu=[0]):
#         pred_set = Dataset_torch(x_pred,with_label=False)
#         data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=batch_size,num_workers=4)
#         trainer = pl.Trainer(gpus=gpu)
#         pred = self.trainer.predict(model=self.model,dataloaders = data_loader_pred)
#         y_pre = torch.tensor([torch.argmax(i) for i in torch.cat(pred)])
#         return y_pre
        