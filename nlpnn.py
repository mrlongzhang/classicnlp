# -*- coding: utf-8 -*-
"""
A hand craft NN NLP model using numpy package.
@author: Zhang Long
"""

from collections import defaultdict

ids=defaultdict(lambda:len(ids))

x='hello my world my friend'
words=x.split(' ')
phi=[None]*len(words)
for word in words:
    index=ids['uni' + word]
    if phi[index] != None:
        phi[index] +=1
    else:
        phi[index] = 1
    print(index)
print(ids)

print(phi)

## perceptron training code (numpy version)
import numpy as np
from collections import defaultdict
import codecs

ftrain_file=".\\data\\titles-en-train.labeled"
ftest_file=".\\data\\titles-en-test.labeled"
phi=[None]*27290
ids=defaultdict(lambda:len(ids))

def create_features(sentence):
    phi=[0]*27290
    words = sentence.split(' ')
    for word in words:
        indx = ids['uni ' + word]
        if indx >= 27290:
            continue
        if phi[indx] != None:
            phi[indx] += 1
        else:
            phi[indx] = 1
    return phi
    
def predict_one(w, phi):
    score = 0
    score = np.dot(w,phi)
    if score >=0:
        return 1
    else:
        return -1

def update_weight(w, phi, y):
    w += np.array(phi)*y
    return w

#ftrain = codecs.open(ftrain_file, 'r', 'utf-8')
#for line in ftrain:
#    line = line.split('\t')
#    sentence = line[1]
#    y=int(line[0])
#    create_features(sentence)
#    
#w=np.zeros(len(ids))
#
#ftrain.seek(0)
#for iter in range(1):
#    for line in ftrain:
#        line = line.split('\t')
#        sentence = line[1]
#        y = int(line[0])
#        phi = create_features(sentence)
#        ypred = predict_one(w, phi)
#        #print('y %d ypred %d' %(y, ypred))
#        if ypred != y:
#            w=update_weight(w, phi, y)
#
#print(w)

############################
# nn forward
# network model : w and b of each layer, it is a list of list.
# the inner list has w as 0th element and b as 1st element
# phi0 input
def forward_nn(network, phi0):
    phi = [phi0]
    for layer in range(1,len(network)+1):
        w=network[layer-1][0]
        b=network[layer-1][1]
        phinu = np.tanh(np.dot(phi[layer-1],w)+b)
        phi.insert(layer,phinu)
    
    return phi

def backward_nn(network, phi, ypred):
    J=len(network)
    lst = ([0]*J)
    lst.append(np.array(ypred-phi[J][0]))
    delta = [np.array([0])]*(J+1)
    delta[J]=np.array(ypred-phi[J][0])
    delta_drv = [np.array([0])]*(J+1)
    
    for i in range(J-1,-1, -1):
        phii1 = phi[i+1]
        onei1 = np.ones_like(phii1)
        phi2 = phii1**2
        delta_drv[i+1] = delta[i+1]*(onei1-phi2)
        w = network[i][0]
        #b = network[i][1]
        delta[i] = np.array(np.dot(w,delta_drv[i+1]))
    
    return delta_drv

def update_weights(network, phi, delta_drv, lmbd):
    for i in range(len(network)):
        w=network[i][0]
        b=network[i][1]
        w+=lmbd*np.outer(phi[i], delta_drv[i+1])
        b+=lmbd*delta_drv[i+1]


def create_network(layers):
    network=[]
    for layernum in range(len(layers)-1):
        wblst = []
        wblst.append(0.2*np.random.rand(layers[layernum],layers[layernum+1])-0.1)
        wblst.append(np.random.rand(layers[layernum+1]))
        network.append(wblst)
    return network

def create_nnfeatures(sentence,size):
    phi=[0]*size
    words = sentence.split(' ')
    senlen = len(words)
    if senlen > size:
        senlen = size
    
    for i in range(senlen):
        if phi[i] != None:
            phi[i] += 1
        else:
            phi[i] = 1
    return phi

nnet = create_network([27290, 20,10,1])
#train
ftrain = codecs.open(ftrain_file, 'r', 'utf-8')
for line in ftrain:
    line = line.split('\t')
    sentence = line[1]
    y=int(line[0])
    phi0 = create_features(sentence)
    ypred=forward_nn(nnet, phi0)
    delta_drv=backward_nn(nnet, ypred, y)
    update_weights(nnet, ypred, delta_drv, 0.1)

print(nnet)
ftest=codecs.open(ftest_file,'r','utf-8')
for line in ftest:
    line=line.split('\t')
    sentence=line[1]
    y=int(line[0])
    phi=create_features(sentence)
    ypred=forward_nn(nnet,phi)
   
    print(y)
    print(ypred[3])
    
def main():
  pass
  # Any code you like

if __name__ == '__main__':
  main()