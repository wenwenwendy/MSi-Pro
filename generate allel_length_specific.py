#used to generate allel_length_specific files
import pandas as pd

df = pd.read_excel (r'/lustre/wmy/Project/data/data_MSi/MiS train data_all.xlsx')
print (df)

allel_list=df['Allele'].drop_duplicates().tolist()
print(allel_list)#return 95allels

length_list=df['Length'].drop_duplicates().tolist()
 #
for i in allel_list:
     print(i)
     for j in length_list:
         if int(j)>=int(8) and int(j)<=int(11):
            df_peptide=[]
            df_peptide=pd.DataFrame(df_peptide)
            print(j)
            df_peptide= df.loc[(df['Allele']==i)&(df['Length']==j)]
            print(df_peptide)
            print(len(df_peptide))
            df_peptide.to_csv('/lustre/wmy/Project/data/data_MSi/'+ str(i)+'_'+str(j)+'.csv',index=False)
