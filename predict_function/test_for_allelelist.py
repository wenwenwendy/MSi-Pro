#test for allelelist
import sys


# alleletxt  =sys.argv[1]#
alleletxt  = '/lustre/wmy/Project/Project-Pan_Allele/predict_function/list_of_alleles.txt'
with open(alleletxt, "r") as f: 
    data = f.read().split('\n')
    #data1 = f.readlines().split('\n') invalid
    print(data)
    #allelelist 
    #print(allelelist)
with open(alleletxt, "r") as f: 
    data2 = f.readlines()
    print(data2) #with/n
with open(alleletxt, "r") as f: 
    data3 = f.read()
    print(data3) #only str as output