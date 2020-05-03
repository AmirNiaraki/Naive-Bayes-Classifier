# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:10:23 2020

@author: niaraki
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
from sklearn.metrics import confusion_matrix
import time
start = time.process_time()




#importing train data and train label
train_data=pd.read_csv("./20newsgroups/train_data.csv", header=None)
train_data.columns=['docId','wordId','count']
train_label=pd.read_csv('./20newsgroups/train_label.csv', header=None)

#importing test data and test label
test_data=pd.read_csv("./20newsgroups/test_data.csv", header=None)
test_data.columns=['docId','wordId','count']
test_label=pd.read_csv('./20newsgroups/test_label.csv', header=None)


Len_train_data=len(train_data)
Len_train_labels=len(train_label)
Newsgroup_size=train_label.max(axis=0).loc[0]
Total_docs=train_label.size
Vocab_size=train_data.max(axis=0).get(key='wordId')

docs_in_class=np.zeros(Newsgroup_size).astype(int)
all_words_in_class=np.zeros(Newsgroup_size).astype(int)
#the last docId that is in class c+1
docId_in_class=np.zeros(Newsgroup_size).astype(int)
prior=np.zeros(Newsgroup_size)
wordId_in_class=np.zeros((Vocab_size, Newsgroup_size))
p_mle=np.zeros((Vocab_size, Newsgroup_size)) 
p_mle_norm=np.zeros((Vocab_size, Newsgroup_size))
p_be=np.zeros((Vocab_size, Newsgroup_size))
p_be_norm=np.zeros((Vocab_size, Newsgroup_size))
omega_nb=np.zeros(Vocab_size)
predict_be=np.zeros(Len_train_labels)
predict_mle=np.zeros(Len_train_labels)
posteriors_be=np.zeros(Newsgroup_size)
posteriors_mle=np.zeros(Newsgroup_size)


#this function calculates class priors and return abs(ln(priors))
def priors(train_label):
    i=1
    iter=0
    
    while i<=Newsgroup_size:
        for row in range(Len_train_labels):
    
            iter+=1
            if row+1==Total_docs:
                docs_in_class[i-1]=iter      
                docId_in_class[i-1]=row+1
                i+=1
                
            elif train_label.at[row,0]!=i:
                docs_in_class[i-1]=iter-1
                docId_in_class[i-1]=row
                iter=1
                i+=1
    
    #calculating the priors for each class
    prior=docs_in_class/Total_docs
    print("Priors for each class: \n", prior)
    prior=abs(np.log(prior))
    return prior
####################################################################

c=0
counter=int(0)
print("\n counting all words in each class:")
#for row in tqdm(range(Len_train_data)):
for row in range(Len_train_data):
    
    wordId_in_class[train_data.at[row, 'wordId']-1,c]+=train_data.at[row,'count']
    if train_data.at[row,'docId']>docId_in_class[c]:
        all_words_in_class[c]=counter
        c+=1
        counter=int(0)
        
    counter+=train_data.at[row,'count']
all_words_in_class[c]=counter

print("\n calculating P_MLE and P_BE:")
#for row in tqdm(range(Vocab_size)):
for row in range(Vocab_size):

    for c in range(Newsgroup_size):
        if wordId_in_class[row,c]>1e-7:
            p_mle[row,c]=wordId_in_class[row,c]/all_words_in_class[c]
            p_be[row,c]=(wordId_in_class[row,c]+1)/(all_words_in_class[c]+Vocab_size)
            
   

Sum_mle=np.sum(p_mle)
Sum_be=np.sum(p_be)

for row in range(Vocab_size):

    for c in range(Newsgroup_size):
        if p_mle[row,c]/Sum_mle>1e-6:
            p_mle_norm[row,c]=abs(np.log(p_mle[row,c]/Sum_mle))
        if p_be[row,c]/Sum_be>1e-6:
            p_be_norm[row,c]=abs(np.log(p_be[row,c]/Sum_be)) 


print("\n Predicting classes for each document via Naive Bayes Classifier:")
d=1
for row in tqdm(range(Len_train_data)):
    
    if  train_data.at[row,'docId']==d+1:
        
        posteriors_be+=prior
        posteriors_mle+=prior
        
        predict_be[d-1]=np.argmax(posteriors_be)+1
        predict_mle[d-1]=np.argmax(posteriors_mle)+1

        posteriors_be=np.zeros(Newsgroup_size)   
        posteriors_mle=np.zeros(Newsgroup_size)          

        if d<Len_train_labels:
            d+=1
#            print("\n now calculating document:", d, "in row", row)
     
    elif train_data.at[row,'docId']==d:
        posteriors_be[:]+=(p_be_norm[train_data.at[row,'wordId']-1,:])*(train_data.at[row,'count'])

        posteriors_mle[:]+=(p_mle_norm[train_data.at[row,'wordId']-1,:])*(train_data.at[row,'count'])

    elif d<Len_train_labels:
                d+=1
                print("document not found", d)            

posteriors_be+=prior
posteriors_mle+=prior
   
predict_be[d-1]=np.argmax(posteriors_be)+1
predict_mle[d-1]=np.argmax(posteriors_mle)+1

c_be=0
c_mle=0                     
for d in range(Len_train_labels):
    if predict_be[d]==train_label.at[d,0]: c_be+=1
    if predict_mle[d]==train_label.at[d,0]: c_mle+=1

accuracy_be=c_be/Len_train_labels
accuracy_mle=c_mle/Len_train_labels
print("\n accuracy of be on Training Data: ", accuracy_be*100)
print("\n accuracy of mle on Training Data: ", accuracy_mle*100)  
     
#Confusion Matrix
print("confusion matrix of /n  be:/n" , confusion_matrix (train_label, predict_be))
print("confusion matrix of /n  mle:/n",  confusion_matrix (train_label, predict_mle))


print("The run time was: ", time.process_time() - start, "seconds")

if __name__=="__main__":
    priors(train_data, train_label)
#    training()
