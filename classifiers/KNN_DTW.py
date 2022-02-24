import torch
import numpy as np
from utils.dtw_cuda import SoftDTW


class Classifier_KNN_DTW:
    def __init__(self):
        pass
    
    def dist(self,x,y):
        return SoftDTW(use_cuda=True)(x,y)
    
    def fit(self, x_train, y_train):
        self.x_train = torch.tensor(x_train).cuda()
        self.y_train = torch.tensor(y_train).cuda()
        self.dtw = SoftDTW(use_cuda=True)

        
    def compute_dtw_one_to_all(self, ts_compare,ts_train_x,index):
  
        ts_compare = ts_compare[index:index+1,:,:].repeat(ts_train_x.shape[0],1,1)

        for i in range(22):

            loss = self.dtw(ts_compare[:,i:i+1,:], ts_train_x[:,i:i+1,:])

            if i == 0:
                dist_list = loss/22
            else:
                dist_list += loss/22
        return dist_list 
    
    def knn_dtw_predict(self, ts_compare, ts_train_x, ts_train_y, index, K):
        
        dist_list = self.compute_dtw_one_to_all(ts_compare,ts_train_x,index=index).cpu()
        ts_train_y = ts_train_y.cpu()

        predict_labels = []

        nearest_series_labels = np.array(ts_train_y[np.argpartition(dist_list, K)[:K]]).astype(int)
        preditc_labels_single = np.argmax(np.bincount(nearest_series_labels))
        predict_labels.append(preditc_labels_single)

        return predict_labels[0]
    
    def predict(self, x_pred, k):
        
        self.x_pred = torch.tensor(x_pred).cuda()
        
        predict_all = []
        
        for i in range(x_pred.shape[0]):
            predict_all.append(self.knn_dtw_predict(self.x_pred,self.x_train,self.y_train,index=i, K=k))

        return np.array(predict_all)