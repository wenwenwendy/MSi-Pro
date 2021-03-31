##
#模型预测,输入测试集,输出预测结果  
# How to load and use weights from a checkpoint
import numpy
from operator import index
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
import tensorflow
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
# used for k fold CV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report


# fix random seed for reproducibility
seed= 7
numpy.random.seed(seed)

# train Model params
nEpochs = 10
batch_size = 30
n_hidden_1 = 50
dim_1D=220

# create model
model = Sequential()
model.add(Dense(n_hidden_1, input_dim=dim_1D, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


####predict 11mers from DFCI-5341 A0301	A310102	B1402 -B3502	C0401	C0802

# load weights
model.load_weights("data/tmpfortest/A0301_11_fold_var1bestweights.h5")

# Compile model
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
print("Created model and loaded weights from A0301 file")

# Compile model (required to make predictions)
# load pima indians dataset
#dataset= numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
def predict(peptides):
    x = [list(p) for p in peptides]
    encoder = OneHotEncoder(
        categories=[list(amino_acids)] * 11)
    encoder.fit(x) 
    return model.predict(encoder.transform(x).toarray()).squeeze() 
    x = [list(p) for p in peptides]
    encoder = OneHotEncoder(
        categories=[list(amino_acids)] * 11)
    encoder.fit(x)
    return model.predict(encoder.transform(x).toarray()).squeeze()

X_test= ['ITQIEHEVSSS',
'LPAAGVGDMVM',
'LPIDVTEGEVI',
'VLAPEGSVPNK',
'VVAPPGVVVSR',
'YPYDGIHPDDL',
'EPLAESITDVL',
'FLYQQQGRLDK',
'GTKQQEIVVSR',
'HPDYGSHIQAL',
'HPEDLQAANRL',
'KLKDQNIFVSR',
'KQRDLEGAVSR',
'KTITGKTFSSR',
'KYLQEEVNINR',
'LPEDEGHTRTL',
'RLSGVSSNIQK',
'RTQLYEYLQNR',
'RVLEKLGVTVR',
'RVQEAVESMVK',
'RVYSPPEWISR',
'VNVEINVAPGK',
'VTNTRTPSSVR',
'KSPASDTYIVF']

y_pred = predict(X_test)
print('the possibility of belonging to A0301:', y_pred)