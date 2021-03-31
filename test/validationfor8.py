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
import tensorflow as tf
#set_random_seed(1231)
tf.random.set_seed(1231)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

from tensorflow.keras.models import load_model

### Load data
import os
os.getcwd() #get current work direction
os.chdir('/lustre/wendy/Project-Pan_Allele/model_trained') #change direction

# 重新创建完全相同的模型，包括其权重和优化程序
new_model = load_model('A0101_8_model.h5',compile = False)

# 显示网络结构
new_model.summary()