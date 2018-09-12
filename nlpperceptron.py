# -*- coding: utf-8 -*-
"""
A perceptron implementation using Python.
@author: Zhang Long
"""
import numpy as np
import codecs

#nlp-perceptron
# phi the feature vector of the sentence the input statement
def create_feature(sentence):
    phi=dict()
    words=sentence.split(' ')
    for word in words:
        if 'u '+word in phi:
            phi['u '+word] += 1
        else:
            phi['u ' +word] = 1
        
    return phi

# w the weight matrix
# y = sign(w.*phi) 
def predict(w, phi):
    score = 0.
    for uword, value in enumerate(phi):
        uword=uword[2:]
        if uword in w:
            score += w[uword]*value
        else:
            w[uword] = 0. # unknown word is regarded as irrelevent
        
    if score >=0:
        return 1
    else:
        return -1
        

# y the label in the train data
# w <-- w +y.*phi
def update_w(w, phi, y):
    for uname, value in enumerate(phi):
        uname = uname[2:]
        # y is 1 or -1 for perceptron
        w[uname] += value*y
        # y is e^(w*phi)/[(1+e^(w*phi))^2] when y = 1
        # or -e^(w*phi)/[(1+e^(w*phi))]^2 when y = 0
        # the updating could be weighted by alpha

# the perceptron algorithm update when predict wrong
def train_perceptron(ftrain_data, margin):
    w={}
    for iter in range(1000):
        ftrain_data = codecs.open(ftrain_data,'r','utf-8')
        for line in ftrain_data:
            line = line.split('\t')
            sentence = line[1]
            y = line[0]
            y = int(y)
            phi=create_feature(sentence)
            ypredict = predict(w, phi)
            # the perceptron algorithm update when predict wrong
            if ypredict != y:
                update_w(w,phi,y)
            # the advance discriminative method 
            # used a margin
            # when margin is 0, it's the perceptron algorithm
            val = w*phi*y
            if val <= margin:
                update_w(w, phi, y)
    return w       

# efficiency trick
# update_weight, predict and create_feature are for a sentence.
# however, regularize weight is for entire weight matrix.
# the trick is to apply reg when the corresponding elements are used.
# thus, you have the getw func

# last dict of weight name:previous iteration that used this term (w[name])
# when to use it?
# in update only
def getw(w, name, c, iter, last):
    if iter != last[name]:
        regterm = c*(iter-last[name])
        if np.abs(w[name]) <= regterm:
            w[name] = 0
        else:
            w[name]-=np.sign(w[name])*regterm
        last[name] = iter
    return w[name]

def main():
  pass
  # Any code you like

if __name__ == '__main__':
  main()