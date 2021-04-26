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

#Create model
model_class = ModelDemo()

###load data##
binder_xlsx =(sys.argv[1])#"/lustre/wmy/Project/data/data_MSi/data_evaluate/DFCI-5283-9.xlsx"
allelelist  =(sys.argv[2:4])#['A0101_9.1.h5','A0201_9.1.h5','B0702_9.1.h5','B0801_9.1.h5','C0701_9.1.h5','C0702_9.1.h5']
#allelelist =[list(p) for p in allelelist]
modelweight_dir = sys.argv[4] #'/lustre/wmy/Project/data/from_Prof.Sun/9mer_k=99/models/'
print('Alleles to be predictied: ',allelelist)

cutoff_Pvalue=0.2
#load DFCI-5283-9 data to evaluate
df_binders=pd.read_excel(binder_xlsx,nrows = 100)
binders=df_binders['Peptide'].tolist()
print('len of binders:',len(binders))


x_pred=[list(p) for p in binders]# to be predicted peptides
#print(x_pred)
y_label=[1] * len(binders) # to be predicted peptides label
#print(y_label)

#allelelist=['A0101_9.1.h5','A0201_9.1.h5','B0702_9.1.h5','B0801_9.1.h5','C0701_9.1.h5','C0702_9.1.h5']


for j in allelelist:
    modelweight=str(modelweight_dir)+j

    y_pred=model_class.predict(x_pred,modelweight)
    #print(y_pred)

    df_ppv=pd.DataFrame({"peptides":binders,"y_label" : y_label, "y_pred" : y_pred.flatten()})
    df_ppv['pred_label'] = ''
    print("df_PPV:",df_ppv)

    #print(df_ppv.loc[df_ppv['y_test']==1])

    #df_ppv_sort=df_ppv.sort_values("y_pred",inplace=False,ascending=False)
    #ascending当传入False时，按照降序进行排列
    #print("sorted data acoording to y_pred:",df_ppv_sort)
    #reindex_df_ppv_sort=df_ppv_sort.reset_index()#add new index
    
    #cutoff=reindex_df_ppv_sort['y_pred'][int(len(binders)-1)]
    #print("cutoff of y_pred:", cutoff)#get cutoff

    #predict_as_binder=reindex_df_ppv_sort.loc['y_pred']
    #df_ppv.loc[df_ppv['y_pred']>cutoff,'predlabel'] = 1
    #df_ppv.predlabel[df_ppv.y_pred>cutoff] = 1
    #df.B[df.A>4] = 0
    #correct_predict=predict_as_binder.loc[:,'y_test'].value_counts()
    #print("numbers of binders been predicted as ture:",correct_predict[1])

    df_ppv['pred_label'] = df_ppv.y_pred.apply(lambda x: 1 if x > cutoff_Pvalue else 0)
    print(df_ppv)
    df_ppv.to_csv('/lustre/wmy/Project/data/data_MSi/data_evaluate/'+'DFCI-5283_'+j+'test_apr26.csv')




