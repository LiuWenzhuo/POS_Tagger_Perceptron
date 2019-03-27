# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:58:32 2019

@author: Lucil
"""

import nltk
import pickle
from perceptron import perceptron

## Download dataset from nltk universal file.
tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
tagged_words = list(set([tup for i in tagged_sentence for tup in i]))
vocab = list(set([word for word, tag in tagged_words]))
tags = list(set([tag for word, tag in tagged_words]))

## Split dataset into training set and test set
#train = tagged_sentence[:int(len(tagged_sentence)*0.7)]
#test = tagged_sentence[int(len(tagged_sentence)*0.7+1):]

alpha = perceptron(tagged_sentence, tags, 50, avg=True)
with open('alpha.pickle', 'wb') as f:
    pickle.dump(alpha, f)
