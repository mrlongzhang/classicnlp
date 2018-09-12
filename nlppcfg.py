# -*- coding: utf-8 -*-
"""
A Probabiltics Context Free Grammer (PCFG) Parser using Python.
This code implemented a weighted graph search
@author: Zhang Long
"""
import codecs
from collections import defaultdict
import math

f_grammer=".\\test\\08-grammar.txt"

nonterm=[]
preterm=defaultdict(list)

grammer_file=codecs.open(f_grammer, 'r','utf-8')
index = 0
for rule in grammer_file:
    words = rule.split('\t')
    lhs = words[0]
    rhs = words[1]
    prob = float(words[2])
    rhs_symbols=rhs.split(' ')
    if len(rhs_symbols) == 1:
        preterm[rhs].append([lhs, math.log(prob)])
    else:
        nonterm.insert(index,[lhs, rhs_symbols[0], rhs_symbols[1],math.log(prob)])

# add pre-terminals 
f_text=".\\test\\08-input.txt"
text_file=codecs.open(f_text, 'r', 'utf-8')

# init best score with lowest level
best_score=defaultdict(lambda: float('-inf'))
best_edge={}
for line in text_file:

    words = line.split(' ')
    
    for i in range(len(words)):
        word = words[i].strip()
        for item in (preterm[word]):
            lhs = item[0]
            log_prob  = item[1]
            ibs = lhs + ' ' + str(i)  + ' ' + str(i+1)
            best_score[ibs] = (log_prob)

text_file.close()
 
#cyk, calculate the rest levels
text_file=codecs.open(f_text,'r','utf-8')
my_lp = float('-inf')
for j in range(2, len(words)+1):
    for i in range(j-2, -1, -1):
        for k in range(i+1, j):
            # rules in grammer table
            for nrul in range(len(nonterm)):
                sym=nonterm[nrul][0] 
                lsym=nonterm[nrul][1] 
                rsym=nonterm[nrul][2] 
                logprob =nonterm[nrul][3]
                ilsym = lsym +' ' + str(i) + ' ' + str(k)
                irsym = rsym +' ' + str(k) + ' ' + str(j)
                
                if best_score[ilsym] > float('-inf') and best_score[irsym] > float('-inf'):
                    my_lp = best_score[ilsym] + best_score[irsym] + logprob
                
                    isymi = sym + ' ' + str(i) + ' ' + str(j)
                    if(my_lp > best_score[isymi]):
                        best_score[isymi] = my_lp
                        best_edge[isymi] = [ilsym,irsym]
                    
def Print(sym, best_edge, words):
    if sym in best_edge:
        symp = sym.split(' ')[0]
        return "("+symp+" " \
            +Print(best_edge[sym][0], best_edge, words) +" " + Print(best_edge[sym][1],best_edge, words) \
            + ")"
    else:
        i = sym.split(' ')[1]
        symp = sym.split(' ')[0]
        return "(" + sym + " " + words[int(i)]+")"

print(Print('S 0 7',best_edge,words))

def main():
  pass
  # Any code you like

if __name__ == '__main__':
  main()