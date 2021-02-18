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
from tensorflow import set_random_seed
set_random_seed(1231)

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
    
    def train(self, binders, nonbinders):
        ### Data prep
        bindersx10 = random.choices(binders, k=10*len(binders))
        nonbindersx10 = random.choices(nonbinders, k=len(bindersx10))
        x = [list(p) for p in bindersx10] + [list(s) for s in nonbindersx10]
        y = [1] * len(bindersx10) + [0] * len(nonbindersx10)
        
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * 9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()
        dim_1D = len(encoder.categories_)*20
        
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
        model.fit(x = encoded_x, y = y,
                  verbose=0,
                  batch_size=batch_size, 
                  epochs=nEpochs, shuffle=True, 
                  validation_split=0.1,
                  class_weight=None, sample_weight=None, initial_epoch=0,
                  callbacks = callbacks)
        
        self.model = model
    
    def predict(self, peptides):
        x = [list(p) for p in peptides]
        encoder = OneHotEncoder(
            categories=[list(amino_acids)] * 9)
        encoder.fit(x)
        return self.model.predict(encoder.transform(x).toarray()).squeeze()
    
### Load data
hits_train = open('peptides_A0101_9.txt', mode='r').readlines()
hits_train = [x.strip() for x in hits_train] 
decoys_train = open('decoys_train_9.txt', mode='r').readlines()
decoys_train = [x.strip() for x in decoys_train] 

### Train model
model = ModelDemo()
# For demo purposes only, this model training is carried
# out on the full data set, no cross-fold splits here!
model.train(hits_train, decoys_train)

### Use model to predict
# For demo purposes only, this prediction is on 
# the same set of hits the model was trained on!
preds = model.predict(hits_train)
