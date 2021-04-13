#!/usr/local/bin/python

import os
import sys
import csv
import numpy as np
import keras
import tensorflow 
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from operator import index
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# used for k fold CV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

# Make sure the model is reproducible
from random import choices
#from numpy.random import seed

seed = 1171
np.random.seed(seed)
seqlen = 8

amino_acids = "ACDEFGHIKLMNPQRSTVWY"


class ModelDemo:
    # Define Model
    def create_model(self, dim_1D, n_hidden_1):
        model = Sequential()
        model.add(Dense(n_hidden_1, input_dim=dim_1D, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    # Define Callbacks
    def get_callbacks(self, patience_lr, patience_es, weight_best_path):
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                           patience=patience_lr, verbose=1, min_delta=1e-3, mode='auto')
        # setup loss value
        #loss_value = 0.001
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience_es,
                                   verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(weight_best_path, monitor='val_accuracy',verbose=1,save_best_only=True, mode='max')
        return [reduce_lr_loss, early_stop,model_checkpoint]

    # verbose = 1 为输出进度条记录
    # verbose = 0 为不在标准输出流输出日志信息

    def get_ppv(self,df_ppv):
        df_ppv.sort_values("y_pred",inplace=True,ascending=False)
        length_top=len(df_ppv)*0.001
        reindex_dfppv=df_ppv.reset_index()
        cut_reindex_dfppv=reindex_dfppv.loc[:int(length_top)]
        counting_cut_reindex_dfppv=cut_reindex_dfppv.loc[:,'y_test'].value_counts()
        defined_ppv=counting_cut_reindex_dfppv[1]/length_top
        return defined_ppv

    def train(self, binders, nonbinders, modelname):

        ###### Data prep ######   
        #used choices generate decoys_train 999 times of hits_train
        nonbinders= choices(nonbinders,k=999*len(binders))

        x = [list(p) for p in binders] + [list(s) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * seqlen)
        encoder.fit(x)
        x = np.array(x)
        y = np.array(y)

        # build a blank dataframe
        #df_report=pd.DataFrame(columns=('precision','recall','f1-score','support'))
        #cvscores = []

        ppvsores= []
        ###### define 5-fold cross validation test harness######
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_var = 1

        for train, test in kfold.split(x, y):
            print('\nFold ',fold_var)
            # train Model params
            nEpochs = 5
            #nEpochs = 1
            #batch_size = 30
            batch_size = 200 

            n_hidden_1 = 50
            # set callback parms
            patience_lr = 2
            # number of epochs to stop without improving 
            patience_es = 1
            weight_best_path =  modelname + "." + str(fold_var) + '.h5'
            #save model at every fold
            #model.save('/lustre/wendy/movebyzx/data/model_trained/'+ modelname +'_fold_var'+ str(fold_var)+".h5")
            callbacks = self.get_callbacks(patience_lr, patience_es,weight_best_path)

            # 在traindata里随机选择，产生10倍大小的traindata
            trainx10 = choices(train, k=10*len(train))

            # onehot squencing
            x_to_train = encoder.transform(x[trainx10]).toarray().astype(int)
            dim_1D = len(encoder.categories_)*20
            # eg.9mers,全连接的输入dim为9*20=180，9是特征的个数，即9个长度，20是20个氨基酸
            y_to_train = np.array(y[trainx10])

            # create model
            model = None
            model = self.create_model(dim_1D, n_hidden_1)
            # Compile model
            model.compile(optimizer='rmsprop',
                          loss='binary_crossentropy', metrics=['accuracy'])

            # Fit the model
            model.fit(x_to_train, y_to_train, verbose=1,
                      batch_size=batch_size,
                      validation_data = (x[test], y[test]),
                      epochs=nEpochs, shuffle=True,
                      class_weight=None, sample_weight=None, initial_epoch=0,
                      callbacks=callbacks)

            # evaluate the model
            #scores = model.evaluate(x[test], y[test], verbose=0)
            #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            #cvscores.append(scores[1] * 100)
            
            # predict the model
            y_pred = model.predict(x[test])
            print(y_pred)
            print(y_pred.flatten())
            #tn,fp,fn,tp=confusion_matrix(y[test],y_pred).ravel()
            #PPV = tp/(tp+fp)
            #print('ppv of ',fold_var,'is:',PPV)
            dic_ppv={"y_test" : y[test], "y_pred" : y_pred.flatten()}
            df_ppv=pd.DataFrame(dic_ppv)
            #df_ppv.to_csv('/lustre/wmy/Project/data/dataframe_ppv/'+ str(modelname) +'_fold_var_'+ str(fold_var))
            ppvsores.append(self.get_ppv(df_ppv))
            #classification report
            #report = classification_report(y[test], y_pred.round(), output_dict=True) 
            #list_report.append(report) 
            #df_transpose = pd.DataFrame(report).transpose()
            #df_report =df_report.append(df_transpose, ignore_index=False)
            #print(classification_report(y[test], y_pred.round()))
            
            #save model at every fold
            #model.save('/lustre/wendy/movebyzx/data/model_trained/'+ modelname +'_fold_var'+ str(fold_var)+".h5")

            fold_var += 1
            #confusion_matrix
            #cm = confusion_matrix(y[test], y_pred.round())
            #tn, fp, fn, tp = cm.ravel()
        #print("classification_report:",df_report)
        #print('cv:',cvscores)
        #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        
        #print(list_report)

        #save reports
        #df_report.to_csv('/lustre/wendy/movebyzx/data/reports/'+ modelname,index=True)
        #data_cvscores = np.array(cvscores)
        #np.savez('/lustre/wendy/movebyzx/data/reports/'+ modelname+'cvscores', data_cvscores)
        #model.summary()
        df_ppvsores=pd.DataFrame(ppvsores)
        df_ppvsores.head()
        df_ppvsores.to_csv(modelname)
        self.model = model

    def predict(self, peptides):
        x = [list(p) for p in peptides]
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * seqlen)
        encoder.fit(x)
        return self.model.predict(encoder.transform(x).toarray()).squeeze()

negdatafile = sys.argv[1]
print('neg data ',negdatafile)
positivedatafile = sys.argv[2]
print('posi dat ',positivedatafile)
modelname = sys.argv[3]
print('modelname ',modelname)

decoys_train = open(negdatafile, mode='r').readlines()
decoys_train = [x.strip() for x in decoys_train]
print('len of decoys_train:',len(decoys_train))


#filename  = "traindata1.csv"
print('now is training:', positivedatafile)

with open(positivedatafile) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        hits_train = [row[2] for row in reader]
        print(hits_train)

hits_train = [x.strip() for x in hits_train]
print('len of hits_train:',len(hits_train))


model = ModelDemo()
model.train(hits_train, decoys_train,modelname)

#


