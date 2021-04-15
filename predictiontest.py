#used for load model weight and predict
#!/usr/local/bin/python
import os
from class_Model import ModelDemo
import csv

# Train model Create modeld
model_class = ModelDemo()
#load A0101_11_fold_var1 model weight for test
hit=['AAPAKKPYRKA', 'AGGVQRQEVVC', 'AIIDPGDSDII','AIMELDDTLKY','ATELDAWLAKY']
probability=model_class.predict(hit,'data/data_MSi/trained models/A0101_11_fold_var5bestweights.h5')
print(probability*100)
#print("the probability of ALDESFLGTLY belong to A0101 : {:5.2f}%".format(100 * probability))