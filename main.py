# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:13:46 2019

@author: Lucil
"""

from perceptron import Viterbi
import pickle

if __name__ == "__main__":
    with open('alpha.pickle', 'rb') as f:
        alpha = pickle.load(f)

    tags =  ['PRON', 'ADV', 'CONJ', 'X', '.', 'ADP', 'DET', 'NUM', 'VERB', 'NOUN', 'ADJ', 'PRT']

    sentence = input("Input a sentence: ")
    sentence = sentence.split(" ")
    print(Viterbi(alpha, sentence, tags))