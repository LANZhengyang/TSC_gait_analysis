import multiprocessing

class GridSearchCV:
    def __init__(self, model,para_list,n_splits=7,n_process=10):
        self.model = model
        self.para_list = list(ParameterGrid(para_list))
        self.n_splits = n_splits
        self.n_process = n_process
        
    def fit(self, x_data_cv, y_data_cv):
        pool = multiprocessing.Pool(processes=self.n_process)
    
        kf = StratifiedKFold(n_splits=n_splits)
        
        self.accuracy_mean_list = []
        self.accuracy_std_list = []
        
        self.accuracy_list = []

        for para in self.para_list:

            accuracy = []
            for train_index, test_index in kf.split(x_data_cv,y_data_cv):
                

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
        
        self.best_accuracy_mean = self.accuracy_mean_list[np.argmax(self.accuracy_mean_list)]
        self.best_accuracy_std = self.accuracy_std_list[np.argmax(self.accuracy_mean_list)]
        self.best_para = self.para_list[np.argmax(self.accuracy_mean_list)]
        print('------------------------')
        print('best parameter:',self.best_para)
        print('accuracy:',self.best_accuracy_mean,"±",self.best_accuracy_std)
        
        self.accuracy_list = np.array(self.accuracy_list)
        