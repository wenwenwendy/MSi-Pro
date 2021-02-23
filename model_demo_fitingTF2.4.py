from operator import index
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
import tensorflow
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
import pandas as pd
# used for k fold CV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
# Make sure the model is reproducible


from random import choices
#from numpy.random import seed
seed = 1171
np.random.seed(seed)


amino_acids = "ACDEFGHIKLMNPQRSTVWY"


class ModelDemo:
    # Define Model
    def create_model(self, dim_1D, n_hidden_1):
        model = Sequential()
        model.add(Dense(n_hidden_1, input_dim=dim_1D, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Define Callbacks
    def get_callbacks(self, patience_lr, patience_es):
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                           patience=patience_lr, verbose=1, min_delta=1e-3, mode='auto')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_es,
                                   verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        return [reduce_lr_loss, early_stop]
    # verbose = 1 为输出进度条记录
    # verbose = 0 为不在标准输出流输出日志信息

    def train(self, binders, nonbinders, modelname):
        # Data prep
        # 在binder里随机选择，产生10倍大小的binder
        bindersx10 = choices(binders, k=10*len(binders))
        nonbindersx10 = choices(nonbinders, k=len(bindersx10))
        x = [list(p) for p in bindersx10] + [list(s) for s in nonbindersx10]
        y = [1] * len(bindersx10) + [0] * len(nonbindersx10)
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * 9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()
        dim_1D = len(encoder.categories_)*20
        # 全连接的输入dim为9*20=180，9是特征的个数，即9个长度，20是20个氨基酸

        # Model params
        nEpochs = 15
        batch_size = 50
        n_hidden_1 = 50
        patience_lr = 2
        patience_es = 4
        callbacks = self.get_callbacks(patience_lr, patience_es)

        # Train model
        x = np.array(encoded_x)
        y = np.array(y)

        # define 5-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # build a blank dataframe
        df_report=pd.DataFrame(columns=('precision','recall','f1-score','support'))

        cvscores = []
        list_report=[]

        fold_var = 1
        save_dir = '/lustre/wendy/data/model_trained/'

        for train, test in kfold.split(x, y):
            # create model
            model = None
            model = self.create_model(dim_1D, n_hidden_1)
            # Compile model
            model.compile(optimizer='rmsprop',
                          loss='binary_crossentropy', metrics=['accuracy'])
            # Fit the model
            model.fit(x[train], y[train], verbose=2,
                      batch_size=batch_size,
                      epochs=nEpochs, shuffle=True,
                      class_weight=None, sample_weight=None, initial_epoch=0,
                      callbacks=callbacks)
            # evaluate the model
            scores = model.evaluate(x[test], y[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)

            # predict the model
            y_pred = model.predict(x[test])
            
            #classification report
            report = classification_report(y[test], y_pred.round(), output_dict=True) 
            #list_report.append(report) 
            df_transpose = pd.DataFrame(report).transpose()
            df_report =df_report.append(df_transpose, ignore_index=False)
            print(classification_report(y[test], y_pred.round()))
            
            #save model at every fold
            model.save('/lustre/wendy/data/model_trained/'+ modelname +'fold_var'+ str(fold_var)+".h5")

            fold_var += 1
            #confusion_matrix
            #cm = confusion_matrix(y[test], y_pred.round())
            #tn, fp, fn, tp = cm.ravel()
        print(df_report)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print(cvscores)
        #print(list_report)
        df_report.to_csv('/lustre/wendy/data/reports/'+ modelname,index=True)
        model.summary()
        self.model = model

    def predict(self, peptides):
        x = [list(p) for p in peptides]
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * 9)
        encoder.fit(x)
        return self.model.predict(encoder.transform(x).toarray()).squeeze()


# Load data
os.getcwd()  # get current work direction
os.chdir('/lustre/wendy/Project-Pan_Allele')  # change direction
hits_train = open('peptides_A0101_9.txt', mode='r').readlines()
hits_train = [x.strip() for x in hits_train]
decoys_train = open('decoys_train_9.txt', mode='r').readlines()
decoys_train = [x.strip() for x in decoys_train]


# Train model
# create modeld
model = ModelDemo()
# For demo purposes only, this model training is carried
# out on the full data set, no cross-fold splits here!

model.train(hits_train, decoys_train, 'testforaddkfold_model')
# Use model to predict
# For demo purposes only, this prediction is on
# the same set of hits the model was trained on!
preds = model.predict(['SADSFASFY'])
print(preds)
