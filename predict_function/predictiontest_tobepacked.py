#used for load model weight and predict
#!/usr/local/bin/python
#seqlen=9; allele type=A0101_9
import os
import sys
import csv
import numpy as np
import pandas as pd
from class_Model import ModelDemo

#from numpy.random import seed
from random import choices
seed = 1171
np.random.seed(seed)


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seq_length', type=int, default = 9)
parser.add_argument('--binder_xlsx', type=str, default = None)
parser.add_argument('--alleletxt', type=str, default=None)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--modelweight_dir', type=str, default=None)
args = parser.parse_args()
print(args.binder_xlsx)
print(args.alleletxt)
print(args.seq_length)

#Create model
model_class = ModelDemo(args.seq_length)

###load data###
binder_xlsx = args.binder_xlsx#load DFCI-5283-9.xlsx data to evaluate
#/lustre/wmy/Project/data/data_MSi/data_evaluate/DFCI-5283-9.xlsx
alleletxt  = args.alleletxt #['A0101_9.1.h5','A0201_9.1.h5','B0702_9.1.h5','B0801_9.1.h5','C0701_9.1.h5','C0702_9.1.h5']
modelweight_dir = args.modelweight_dir #/lustre/wmy/Project/data/from_Prof.Sun/9mer_k=99/models/
savedir = args.savedir#/lustre/wmy/Project/Project-Pan_Allele/predict_function/

with open(alleletxt, "r") as f: 
    allelelist = f.read().split('\n')
    #print(allelelist)
print('Alleles to be predictied: ',allelelist)

cutoff_Pvalue=0.2

df_binders=pd.read_excel(binder_xlsx)# for test,nrows = 100
binders=df_binders['Peptide'].tolist()  ##generate list of peptides
print('len of binders:',len(binders))


x_pred=[list(p) for p in binders]# to be predicted peptides
#print(x_pred)
y_label=[1] * len(binders) # to be predicted peptides label
#print(y_label)


for j in allelelist:
    modelweight=str(modelweight_dir)+j+'.4.h5' #use the firth weight of the 5fold CV
    #allelelist=['A0101_9.1.h5','A0201_9.1.h5','B0702_9.1.h5','B0801_9.1.h5','C0701_9.1.h5','C0702_9.1.h5']
    y_pred=model_class.predict(x_pred,modelweight)
    #print(y_pred)

    df_ppv=pd.DataFrame({"peptides":binders,"y_label" : y_label, "y_pred" : y_pred.flatten()})
    df_ppv['pred_label'] = ''
    print("df_PPV:",df_ppv)

    df_ppv['pred_label'] = df_ppv.y_pred.apply(lambda x: 1 if x > cutoff_Pvalue else 0)
    print(df_ppv)
    df_ppv.to_csv(str(savedir)+j+'.csv')




