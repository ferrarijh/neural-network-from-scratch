# -*- coding: utf-8 -*-
"""
by jonathan hahn

script to test prediction accuracy of simple NN, predicting adding network.
x[0]+x[1] = y_test[0], x[2]+x[3] = y_test[1]

This script is just for studying and testing NN.
!! Range of x should not be too large or NN will encounter overflow !!
"""

lr = 0.000001    #learning rate

import matplotlib.pyplot as plt
import numpy as np

def gen():
    global x, y_test
    x = np.random.randint(1, 200, 4)    #harder to predict as range expands
    y_test = np.array([x[0]+x[1], x[2]+x[3]])
#ideal output : o2[0] = x[0]+x[1], o2[1] = x[2]+x[3]

h_nodes = 20    #number of hidden nodes
w1 = np.random.randn(5, h_nodes)    #w1[4] is bias
net1 = []
o1 = []
w2 = np.random.randn(h_nodes+1, 2)      #w2[h_nodes] is bias
net2 = []
o2 = []     #prediction - compare with y_test

def sig(x):
    return 1/(1+np.exp(-x))
def dSig(x):
    return sig(x)*(1-sig(x))

def relu(x):
    if x<0:
        return 0
    else:
        return x
def dRelu(x):
    if x<0:
        return 0
    else:
        return 1

#leaky relu
def myRelu(x):
    if x<0:
        return -0.01*x
    else:
        return x
def dMyRelu(x):
    if x<0:
        return -0.01
    else:
        return 1

def forward():
    global net1, o1, net2, o2
    net1 = np.matmul(x, w1[:len(w1)-1])
    for i in range(len(net1)):
        net1[i]+=w1[len(w1)-1][i]
    o1 = net1
    for i in range(len(o1)):
        o1[i] = myRelu(o1[i])
    net2 = np.matmul(o1, w2[:len(w2)-1])
    for i in range(len(net2)):
        net2[i] += w2[len(w2)-1][i]
    o2 = net2
    for i in range(len(o2)):
        o2[i] = myRelu(o2[i])

def backProp():   #back propagation
    for i in range(len(w2)):    #ith node
        for j in range(len(w2[i])): #jth link
            if i==(len(w2)-1):  #bias
                w2[i][j] += lr*dMyRelu(net2[j])*(y_test[j]-o2[j])
            else:
                w2[i][j] += lr*o1[i]*dMyRelu(net2[j])*(y_test[j] - o2[j])
    forward()
    for i in range(len(w1)):
        for j in range(len(w1[i])):
            if i==(len(w1)-1):
                tmpsum = 0
                for k in range(len(w2[j])):
                    tmpsum += w2[j][k]*dMyRelu(net2[k])*(y_test[k] - o2[k])
                w1[i][j] += lr*dMyRelu(net1[j])*tmpsum
                    
            else:
                tmpsum = 0
                for k in range(len(w2[j])):
                    tmpsum += w2[j][k]*dMyRelu(net2[k])*(y_test[k] - o2[k])
                w1[i][j] += lr*x[i]*dMyRelu(net1[j])*tmpsum

def train():
    gen()
    forward()
    backProp()

def trains(n):
    for i in range(n):
        train()

def loss():
    return ((y_test-o2)**2 /2).sum()
        
#===== plot error function block
        
start = 0
step = 20   #training batch size
end = 5000  #total number of trains
n_trains = np.arange(start, end, step, dtype='int')
n_loss = np.zeros(len(n_trains));
valid_size = 100    #for error testing - NOT training batch
lsum = 0
for i in range(valid_size):
    gen()
    forward()
    lsum += loss()
    
n_loss[0] = lsum/valid_size
for idx in range(1, len(n_loss)):
    trains(step)
    lsum = 0
    for i in range(valid_size):
        gen()
        forward()
        lsum += loss()
    n_loss[idx] = lsum/valid_size

plt.figure(figsize=[12, 6])
plt.plot(n_trains[1:], n_loss[1:], marker='o')
plt.xlabel('number of trains')
plt.ylabel('mean error function')
plt.title('Mean Error Functon Value by Number of trainings')
plt.show()

#======== for further examination ========

def test(): #Test validity
    gen()
    forward()
    print("prediction \t: ",end='')
    print(o2)
    print("prediction(int)\t: ",end='')
    print((o2+0.5).astype(int))
    print("answer \t\t: ",end='')
    print(y_test)

def showME(n): #Show Mean Error. Lower is better.
    lsum = 0
    for i in range(n):
        gen()
        forward()
        lsum += loss()
    print("mean E(o2) of "+str(n)+" trials : "+str(lsum/n))

def hitRate(n):  #Test hit rate. Higher is better.
    hit = 0
    for i in range(n):
        gen()
        forward()
        if ((o2+0.5).astype(int) == y_test).all():
            hit += 1
    print("Trials\t:"+str(n))
    print("Hits\t:"+str(hit))
    print("Accuracy:",end='')
    print(hit/n)

def trainTill(_loss):    #train til loss() avg < 'loss'.
    lavg = 0
    cnt=0
    for i in range(100):
        gen()
        forward()
        lavg += loss()
    lavg /= 100
    while(lavg>_loss):
        trains(100)
        lavg = 0
        for i in range(100):
            gen()
            forward()
            lavg += loss()
        lavg /= 100
        cnt += 100
    print(">>...after "+str(cnt)+" trains.")
    return cnt
    
