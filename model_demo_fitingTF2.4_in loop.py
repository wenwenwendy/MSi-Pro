import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
# Make sure the model is reproducible
from random import choices
from numpy.random import seed
seed(1171)
import tensorflow 
#set_random_seed(1231)
tensorflow.random.set_seed(1231)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

class ModelDemo:
    ### Define Model
    def create_model(self, dim_1D, n_hidden_1):
        model = Sequential()
        model.add(Dense(n_hidden_1, input_dim=dim_1D, activation='relu'))        
        model.add(Dense(1, activation='sigmoid')) 
        return model
    
    ### Define Callbacks
    def get_callbacks(self, patience_lr, patience_es):
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, 
            patience=patience_lr, verbose=1, min_delta=1e-3, mode='auto')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_es, 
            verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        return [reduce_lr_loss, early_stop]
    #verbose = 1 为输出进度条记录
    #verbose = 0 为不在标准输出流输出日志信息
    #for 
    def train(self, binders, nonbinders,modelname):
        ### Data prep
        bindersx10 = choices(binders, k=10*len(binders))#在binder里随机选择，产生10倍大小的binder
        nonbindersx10 = choices(nonbinders, k=len(bindersx10))
        x = [list(p) for p in bindersx10] + [list(s) for s in nonbindersx10]
        y = [1] * len(bindersx10) + [0] * len(nonbindersx10)
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * 8)
        encoder.fit(x)
        #ct = ColumnTransformer(
        #[('one_hot_encoder', OneHotEncoder(categories='auto'), [list(amino_acids)] * 9])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        #remainder='passthrough' )                                        # Leave the rest of the columns untouched)
        #x = ct.fit_transform(x)
        encoded_x = encoder.transform(x).toarray()
        dim_1D = len(encoder.categories_)*20
        #全连接的输入dim为9*20=180，9是特征的个数，即9个长度，20是20个氨基酸
        
        ### Model params
        nEpochs = 15 
        batch_size = 50 
        n_hidden_1 = 50
        patience_lr = 2
        patience_es = 4
        callbacks = self.get_callbacks(patience_lr, patience_es)
        
        ### Train model
        model = None
        model = self.create_model(dim_1D, n_hidden_1)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
        model.fit(x = np.array(encoded_x), y = np.array(y),
                  verbose=2,
                  batch_size=batch_size, 
                  epochs=nEpochs, shuffle=True, 
                  validation_split=0.1,
                  class_weight=None, sample_weight=None, initial_epoch=0,
                  callbacks = callbacks)
        model.save(modelname)
        self.model = model
    
        
    def predict(self, peptides):
        x = [list(p) for p in peptides]
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * 9)
        encoder.fit(x)
        return self.model.predict(encoder.transform(x).toarray()).squeeze()
    
### Load data
import os
os.getcwd() #get current work direction
os.chdir('/home/wmy/SSh_Project/Project-Pan_Allele') #change direction

import os
path="/home/wmy/SSh_Project/Project-Pan_Allele/rawdata/txtfile/8-length/"  #待读取的文件夹
path_list=os.listdir(path)
path_list.sort() #对读取的路径进行排序
decoys_train = open('/home/wmy/SSh_Project/Project-Pan_Allele/rawdata/txtfile/decoys_train_8.txt', mode='r').readlines()
decoys_train = [x.strip() for x in decoys_train]
for filename in path_list:
    #print(os.path.join(path,filename))
    print('now is training:',filename)

    ###set 8-length loop
    hits_train = open(os.path.join(path,filename), mode='r').readlines()
    hits_train = [x.strip() for x in hits_train]
    
    ### Train model
    model = ModelDemo()
    # For demo purposes only, this model training is carried
    # out on the full data set, no cross-fold splits here!
    model.train(hits_train, decoys_train,str(filename[:7]+'_model.h5'))

### Use model to predict
# For demo purposes only, this prediction is on 
# the same set of hits the model was trained on!
#preds = model.predict(['SADSFQSFY'])
#print(preds)

#save model as HDF5 format
