# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:50:51 2019

@author: Lucil
"""


from collections import Counter


## Define feature extractor to create local feature vector phi:
def feature_extractor(sentence, t_prev, t_pres, index):
    dic = {}
    word = sentence[index]
    dic[(word, t_pres)] = 1
    if index == 0:
        dic[('start', t_pres)] = 1
    else:
        dic[(t_prev, t_pres)] = 1
        dic[('pre_w '+sentence[index - 1], t_pres)] = 1
    if index <len(sentence)-1:
        dic[('aft_w '+sentence[index + 1], t_pres)] = 1
    else:
        dic[(t_pres, 'end')] = 1
    return dic

## Define global feature_vector function:
def feature_vector(sentence, tag):
    for j in range(len(sentence)):
        if j == 0:
            v = feature_extractor(sentence,'start',tag[j],j)
        else:
            v = Counter(v) +Counter(feature_extractor(sentence, tag[j-1], tag[j], j))
            v = dict(v)
    return v

def product(alpha, f):
    s = 0
    for key in f.keys():
        if key in alpha.keys():
            s = s+ alpha[key]*f[key]
    return s

##Viterbi algorithm
def Viterbi(alpha, sentence, tags):
    n = len(sentence)
    score = [0]*n
    state = [0]*n
    y = [0]*n
    curr_state_score = {}
    curr_state = {}
    last_state_score = {}
    
    for tag_ in tags:
        curr_state_score[tag_] = product(alpha, feature_extractor(sentence, 'start', tag_, 0))
        curr_state[tag_] = 'start'
    score[0] = curr_state_score.copy()
    state[0] = curr_state.copy()
    for i in range(1, n):
        last_state_score = curr_state_score.copy()
        for tag_c in tags:
            dic = {}
            for tag_l in tags:
                dic[tag_l] = last_state_score[tag_l]+product(alpha, feature_extractor(sentence, tag_l, tag_c, i))
            curr_state_score[tag_c] = max(dic.values())
            curr_state[tag_c] = max(dic, key=dic.get)
        score[i] = curr_state_score.copy()
        state[i] = curr_state.copy()
    y[-1] = max(curr_state_score, key = curr_state_score.get)
    for i in reversed(range(n-1)):
        curr_state = state[i+1]
        y[i] =  curr_state[y[i+1]]
    return y

## Perceptron algorithm
def perceptron(training_set, tags, T, avg = False):
    n = len(training_set)
    alpha = {}
    l = []
    for t in range(T):
        for i in range(n):
            tagged_s = training_set[i]
            tag = [tup[1] for tup in tagged_s ]
            sentence = [tup[0] for tup in tagged_s]
            z = Viterbi(alpha, sentence, tags)
            #print (z)
            if z != tag:
                alpha = Counter(alpha) + Counter(feature_vector(sentence, tag)) - Counter(feature_vector(sentence, z))
                alpha = dict(alpha)
                if avg == True:
                    l.append(alpha)
    if avg == True:
        for al in l:
            alpha = Counter(alpha) + Counter(al)
        alpha = dict(alpha)
        alpha = {k: v /(T*n) for k, v in alpha.items()}
    return alpha