#path="/lustre/wmy/Project/data/rawdata/txtfile/11-length/"
import os
from class_Model import ModelDemo
import csv

########## Load data  ##########
os.getcwd()  # get current work direction
os.chdir('/lustre/wmy/Project/data/data_MSi/')  # change direction
#待读取的文件夹
decoys_train = open('decoys_train_11.txt', mode='r').readlines()
decoys_train = [x.strip() for x in decoys_train]
print('len of decoys_train:',len(decoys_train))
path="/lustre/wmy/Project/data/data_MSi/11_length/"
path_list=os.listdir(path)
path_list.sort() #对读取的路径进行排序

#loop

#print(os.path.join(path,filename))
#print('now is training:',filename)
with open("/lustre/wmy/Project/data/data_MSi/11_length/B1301_11.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    hits_train = [row[2] for row in reader]
    print(hits_train)
hits_train = [x.strip() for x in hits_train]
print(hits_train)
print('len of hits_train:',len(hits_train))
#print(hits_train)
# Train model Create modeld
model = ModelDemo()
model.train(hits_train, decoys_train,'B1301_11')

#


