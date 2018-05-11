#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:35:31 2018

@author: shashi
"""


import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

print("Using tanh Activation function")
print()
print()

print("Preprocessing the input email file")
print()

#remove = dict.fromkeys(map(ord, string.punctuation))
with open('Assignment_2_data.txt') as inputfile:
    f = inputfile.read()


new_f = ""
for char in f:
   if char not in punctuation:
      new_f = new_f + char


g = new_f.splitlines()




l =[]
for i in range(len(g)):
    temp = word_tokenize(g[i])
    l.append(temp)
    



for i in range(len(l)):
    for j in range(len(l[i])):
        l[i][j] = l[i][j].lower()

stop_words = set(stopwords.words("english"))


for i in range(len(l)):
    j=0
    t = len(l[i])
    while j < t:
        if(len(l[i][j]) == 1 or l[i][j].isnumeric()):
             l[i].pop(j)
             t = t-1
        j +=1


for i in range(len(l)):
    j=0
    t = len(l[i])
    while j < t:
        if(l[i][j] in stop_words):
             l[i].pop(j)
             t = t-1
        j +=1
     
output = np.ones(len(l))
for i in range(len(l)):
    if(l[i][0] == 'ham'):
        output[i] = 0
    else:
        output[i] = 1
        
for i in range(len(l)):
    l[i].pop(0)
             
             
ps = PorterStemmer()

for i in range(len(l)):
    j=0
    for j in range(len(l[i])):
         l[i][j] = ps.stem(l[i][j])
         
wordlist = {}
for i in range(len(l)):
    for j in range(len(l[i])):
        if(l[i][j] in wordlist):
             wordlist[l[i][j]] =  wordlist[l[i][j]] + 1
        else :
             wordlist[l[i][j]] = 1

freq_l = sorted(wordlist.values())
freq_l.reverse()

freq_l[1999]             

new_wordlist = {}
new_wordlistt =[]
for key, value in wordlist.items():
    if(wordlist[key] >= freq_l[1999]):
        new_wordlist[key] = value
        new_wordlistt.append(key)
         
for i in range(len(l)):
    j=0
    t = len(l[i])
    while j < t:
        if( l[i][j] not in new_wordlist):
             l[i].pop(j)
             t = t-1
        j +=1

input_mat = np.zeros((len(l), len(new_wordlist)+1),)


for i in range(len(l)):
    input_mat[i][0] = 1

for i in range(len(l)):
    j=0
    for j in range(len(new_wordlistt)):
        flag =0
        k=0
        for k in range(len(l[i])):
            if(new_wordlistt[j] == l[i][k]):
                flag =1
                break
        if(flag == 1):
            input_mat[i][j+1] = 1
        else:
            input_mat[i][j+1] = 0

train_len = (int)(len(l)*0.8)
train_input = input_mat[:train_len, :]
test_input = input_mat[train_len:,:]

train_output = output[:train_len]
test_output = output[train_len:]


            
w12  = 0.01*np.random.uniform(low =-1, high =1, size=(100,2486))
w23  = 0.01*np.random.uniform(low =-1, high =1, size=(50,101))
w34  = 0.01*np.random.uniform(low =-1, high =1, size=(1,51))


#Neural network architecture
 
#input layer --->           layer2 ----->     layer3 ----->  outputlayer
#(2485 feattures + 1 bias)  (100 + 1 bias)    (50 + 1 bias)   (1)

#w12 : (2486*100) weights of edges from input layer to layer2
#w23 : (101*50) weights of edges from layer2 to layer 3
#w34 : (51*1) weights of edges from layer3 to output layer i.e layer4

#sum2 : w12*inputlayer + b (100*1 matrix)
#out2 : tanh(sum2) + 1 is concatenated  (101*1 matrix)   
    
#sum3 : w23*out2 + b (50*1 matrix)
#out3 : tanh(sum3) + 1 is concatenated  (51*1 matrix)   

#sum4 : w34*out3 + b (1*1 matrix)
#out4 : tanh(sum4)  (1*1 matrix)
    
#error_in = error into a layer
#error_out = error a layer projects out to layers behind it 








def tanh(x):
    result = (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
    return np.matrix(result)
    

def tanh_derivative(x):
    return (1-np.square(x))
    


def forward_pass(input_mat, w12, w23, w34, i):
    
    inp = input_mat[i] 
    inp = np.reshape(inp, (2486,1))
    sum2 = np.matmul(w12, inp) #100*1
#    print (sum2.shape)
#    sum2 = np.reshape(sum2,(100,1))
    out2 = tanh(sum2) #100*1
    ones = np.ones((out2.shape[1], 1))
    out2 = np.concatenate((ones, out2), axis=0) #101*1
    
    
    sum3 = np.matmul(w23, out2) #50*1
    out3 = tanh(sum3) # 50*1
    out3= np.concatenate((ones, out3), axis=0) #51*1
    
    
    sum4 = np.matmul(w34, out3) #1*1
    out4 = tanh(sum4)#1*1
    
    return sum2, out2, sum3, out3, sum4, out4

    
def backward_pass(out2, out3, out4, sum2, sum3, sum4, output, input_mat, w12, w23, w34, k, alpha):
    

    error4_out = -2*(output[k] - out4)*tanh_derivative(out4)
    
    
    
#    new_out3 = out3[1:,:]
    grad34 = np.matmul(error4_out, out3.transpose()) # (1*1 ** (51*1)t )= 1*51
    new_w34 = w34[:, 1:]
    error3_in = np.matmul(new_w34.transpose() , error4_out)
    out3_p = out3[1:,:]
    error3_out = np.multiply(tanh_derivative(out3_p), error3_in)
    
    grad23 = np.matmul(error3_out, out2.transpose())
    new_w23 = w23[:,1:]
    error2_in = np.matmul(new_w23.transpose() , error3_out)
    out2_p = out2[1:,:]
    error2_out = np.multiply(tanh_derivative(out2_p), error2_in)
    
    inp = input_mat[0]
    inp = np.reshape(inp, (2486,1))
    grad12 = np.matmul(error2_out, inp.transpose())

    w12 = w12 - alpha*grad12
    w23 = w23 - alpha*grad23
    w34 = w34 - alpha*grad34
    
    
    
    return w12, w23, w34    






epochs = 10
alpha = 0.1
error_sig = np.zeros((epochs,2))
for i in range(epochs):
    j=0
    for j in range(train_input.shape[0]):
        sum2, out2, sum3, out3, sum4, out4 = forward_pass(train_input, w12, w23, w34, j)
        w12, w23, w34 = backward_pass(out2, out3, out4, sum2, sum3, sum4, train_output, train_input, w12, w23, w34, j, alpha)
    
    train_error = 0
    test_error = 0
    m=0
    for m in range(train_input.shape[0]):
        _, _, _, _, _, out4 = forward_pass(train_input, w12, w23, w34, m)
        train_error =  train_error + (out4 - train_output[m])**2
        
    train_error = train_error/(2*train_input.shape[0])
    
    m=0
    for m in range(test_input.shape[0]):
        _, _, _, _, _, out4 = forward_pass(test_input, w12, w23, w34, m)
        test_error =  test_error + (out4 - test_output[m])**2
        
    test_error = test_error/(2*test_input.shape[0])
    
    error_sig[i][0] = train_error
    error_sig[i][1] = test_error
    print("epoch = ", i ," in sample error = ", train_error, "out sample error = ", test_error)
    
a = error_sig[:,0]
b = error_sig[:,1]
plt.title('epochs vs error ')
plt.plot(range(epochs),a, 'g', label='In-sample error')
plt.plot(range(epochs),b, 'b', label='Out of sample error')
plt.ylabel('Mean Suared Error')
plt.xlabel('epochs')
#plt.grid() # grid on
plt.legend() 
plt.show()
    
threshold  = np.linspace(-0.9, 0.9, 19)
for i in range(threshold.shape[0]):
    m=0
    error_count =0 
    for m in range(test_input.shape[0]):
        _, _, _, _, _, out4 = forward_pass(test_input, w12, w23, w34, m)
        if(out4 >= threshold[i]):
            out4 = 1
        else:
            out4 = 0
            
        if(test_output[m] != out4):
            error_count = error_count + 1
            
    print("% Error at threshold = ", threshold[i], "  is ", (error_count/test_input.shape[0])*100)
    
print("Since we get least number of wrong predictions in case of threshold = 0.5 So we choose 0.5 as threshold")

sys.exit()
    
