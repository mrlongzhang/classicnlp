# -*- coding: utf-8 -*-
"""
A unigram language model using Python.
@author: Zhang Long
"""
import codecs
import math

train_file=".\\data\\wiki-en-train.word"
testtrain_file=".\\data\\wiki-en-test.txt"
testtest_file=".\\test\\01-test-input.txt"

# ex 1 create train-unigram
# test-unigram

def train_unigram(file_name):
    wordmap={}
    model={}
    wordcount = 0
    ftrain=codecs.open(file_name,'r', 'utf-8')
    for line in ftrain:
        #line.append(' </s>')
        words=line.split(' ')
        words.append('</s>')
        
        for word in words:
            if word in wordmap:
                wordmap[word] += 1
            else:
                wordmap[word] =1
                
            wordcount +=1
        
    for word in wordmap:
        model[word] = float(wordmap[word])/float(wordcount)
    
    return model

def smothingP(model, lambdaUnknown, N):
    for word in model:
        model[word] = model[word]*(1-lambdaUnknown) + (lambdaUnknown)/float(N)
    return model

def test_model(model,file_name):
    model = smothingP(model, 0.05, 1000000)
    wordcount = 0
    unknown = 0
    entropy = 1.0
    ftest=codecs.open(file_name,'r','utf-8')
    for line in ftest:
        line.strip('\n')
        words=line.split(' ')
        words.append('<\s>')
        
        for word in words:
            if word in model:
                entropy+=-math.log2(model[word])
            else:
                unknown += 1
            wordcount += 1
    
    entropy /= wordcount
    coverage = float(wordcount - unknown)/float(wordcount)
    return entropy, coverage

model = train_unigram(train_file)
print(model)
entropy, coverage = test_model(model, testtrain_file)        
print('entropy is %f coverage is %f' %(entropy,coverage))

def main():
  pass
  # Any code you like

if __name__ == '__main__':
  main()