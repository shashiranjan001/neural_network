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

print("Using Softmax(outerlayer)  sigmoid(inner) Activation function")
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

new_output = np.ones((output.shape[0],2))             
for i in range(output.shape[0]):
    if(output[i] == 1):
        new_output[i][0] = 1
        new_output[i][1] = 0
        
    else:
        new_output[i][0] = 0
        new_output[i][1] = 1

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

train_output = new_output[:train_len,:]
test_output = new_output[train_len:,:]


            
w12  = np.random.uniform(low =-1, high =1, size=(100,2486))
w23  = np.random.uniform(low =-1, high =1, size=(50,101))
w34  = np.random.uniform(low =-1, high =1, size=(2,51))




def sigmoid(x):
    result = 1/(1 + np.exp(-x))
    return np.matrix(result)

def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), (1-sigmoid(x)))    

def softmax(X):
    
    if(X[0][0] > X[1][0]):
        X[1][0] = X[1][0] - X[0][0]
        X[0][0] = 0
        
    else:
        X[0][0] = X[0][0] - X[1][0]
        X[1][0] = 0
        
    rtr= np.ones((2,1))
    temp_sum = np.exp(X[0][0]) + np.exp(X[1][0])
    X_exp1 = np.exp(X[0][0])
    X_exp2 = np.exp(X[1][0])
    rtr[0][0] = X_exp1/temp_sum
    rtr[1][0] = X_exp2/temp_sum

    return rtr

def softmax_derivative(x):
    return np.multiply(x, (1-x))

#Neural network architecture
 
#input layer --->           layer2 ----->     layer3 ----->  outputlayer
#(2485 feattures + 1 bias)  (100 + 1 bias)    (50 + 1 bias)   (1)

#w12 : (2486*100) weights of edges from input layer to layer2
#w23 : (101*50) weights of edges from layer2 to layer 3
#w34 : (51*2) weights of edges from layer3 to output layer i.e layer4

#sum2 : w12*inputlayer + b (100*1 matrix)
#out2 : sigmoid(sum2) + 1 is concatenated  (101*1 matrix)   
    
#sum3 : w23*out2 + b (50*1 matrix)
#out3 : sigmoid(sum3) + 1 is concatenated  (51*1 matrix)   

#sum4 : w34*out3 + b (2*1 matrix)
#out4 : softmax(sum4)  (2*1 matrix)
    
#error_in = error into a layer
#error_out = error a layer projects out to layers behind it 

def forward_pass(input_mat, w12, w23, w34, i):
    
    inp = input_mat[i] 
    inp = np.reshape(inp, (2486,1))
    sum2 = np.matmul(w12, inp) #100*1
#    print (sum2.shape)
#    sum2 = np.reshape(sum2,(100,1))
    out2 = sigmoid(sum2) #100*1
    ones = np.ones((out2.shape[1], 1))
    out2 = np.concatenate((ones, out2), axis=0) #101*1
    
    
    sum3 = np.matmul(w23, out2) #50*1
    out3 = sigmoid(sum3) # 50*1
    out3= np.concatenate((ones, out3), axis=0) #51*1
    
    
    sum4 = np.matmul(w34, out3) #1*1
    out4 = softmax(sum4)
    
    return sum2, out2, sum3, out3, sum4, out4


    
    
    
def backward_pass(out2, out3, out4, sum2, sum3, sum4, output, input_mat, w12, w23, w34, k, alpha):
    
    
#    if(sum4[0][0] > sum4[1][0]):
#        final_out = 0
#    else:
#        final_out = 1
#        
#    error4_out = -2*(output[k] - final_out)*softmax_derivative(out4)
    

    error4_in = -2*(np.reshape(output[k], (2,1)) - out4)
    temp = out4[0][0]*out4[1][0]
    error4_out =  2*temp*error4_in
    
#    new_out3 = out3[1:,:]
    grad34 = np.matmul(error4_out, out3.transpose()) # (1*1 ** (51*1)t )= 1*51
    new_w34 = w34[:, 1:]
    error3_in = np.matmul(new_w34.transpose() , error4_out)
    out3_p = out3[1:,:]
    error3_out = np.multiply(sigmoid_derivative(out3_p), error3_in)
    
    grad23 = np.matmul(error3_out, out2.transpose())
    new_w23 = w23[:,1:]
    error2_in = np.matmul(new_w23.transpose() , error3_out)
    out2_p = out2[1:,:]
    error2_out = np.multiply(sigmoid_derivative(out2_p), error2_in)
    
    inp = input_mat[k]
    inp = np.reshape(inp, (2486,1))
    grad12 = np.matmul(error2_out, inp.transpose())

    w12 = w12 - alpha*grad12
    w23 = w23 - alpha*grad23
    w34 = w34 - alpha*grad34
    
   # if(k == 1):
    #    print (error4_out.shape,error4_in.shape,error3_out.shape,error3_in.shape,error2_out.shape ,error2_in.shape)
    
    
    return w12, w23, w34    



print("Using Softmax Activation function")

epochs = 10
alpha = 0.1
error_sig = np.zeros((epochs,2))
for i in range(epochs):
    
    for j in range(train_input.shape[0]):
        sum2, out2, sum3, out3, sum4, out4 = forward_pass(train_input, w12, w23, w34, j)
        w12, w23, w34 = backward_pass(out2, out3, out4, sum2, sum3, sum4, train_output, train_input, w12, w23, w34, j, alpha)

    train_error = np.zeros((2,1))
    test_error = np.zeros((2,1))
    
    for m in range(train_input.shape[0]):
        _, _, _, _, _, out4 = forward_pass(train_input, w12, w23, w34, m)    
        train_error = train_error +  np.square(out4 - train_output[m].reshape(2,1))
    
    train_error = train_error/train_input.shape[0]
    
    for m in range(test_input.shape[0]):
        _, _, _, _, _, out4 = forward_pass(test_input, w12, w23, w34, m)    
        test_error = test_error +  np.square(out4 - test_output[m].reshape(2,1))
        
    test_error = test_error/train_input.shape[0]    
    error_sig[i][0] = (train_error[0][0]+train_error[1][0])/2
    error_sig[i][1] = (test_error[0][0]+test_error[1][0])/2
    print("epoch = ", i ," in sample error = ", error_sig[i][0], "out sample error = ", error_sig[i][1])
    
    
#for i in range(epochs):
#    
#    
#    
#error_sig = error_sig[::-1]
#error_sig_copy = copy.deepcopy(error_sig)
#for i in range((int)(epochs/2)):
#    temp = error_sig_copy[i]
#    error_sig[i] = error_sig_copy[epochs-1-i]
#    error_sig[epochs-1-i] = temp
    

print(error_sig)
a = error_sig[:,0]
b = error_sig[:,1]
plt.title('Error vs epochs')
plt.plot(range(epochs),a, 'r', label='in sample error')
plt.plot(range(epochs),b, 'b', label='out sample error')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend() 
plt.show()


error_count = 0
for m in range(test_input.shape[0]):
    _, _, _, _, _, out4 = forward_pass(test_input, w12, w23, w34, m)    
    if(out4[0][0]>out4[1][0]):
        temp_out = 1
    else:
        temp_out =0
    if(temp_out != test_output[m][0]):
        error_count = error_count+1
        
        
print("Error % = ", (error_count/test_input.shape[0])*100)
sys.exit()
