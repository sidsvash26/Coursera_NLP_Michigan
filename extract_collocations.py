# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:44:59 2016

@author: sidvash
"""

from collections import defaultdict
import math
import nltk
from nltk.util import ngrams
import random

# Constants to be used to fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    #Empty count dicts    
    unigram_c = defaultdict(int)
    bigram_c = defaultdict(int)
    trigram_c = defaultdict(int)
    quadgram_c = defaultdict(int)
    
    for sentence in training_corpus:
        token0 = sentence.strip().split()
        token1 = token0 + [STOP_SYMBOL]
        token2 = [START_SYMBOL] + token0 + [STOP_SYMBOL]
        token3 = [START_SYMBOL] +[START_SYMBOL]+ token0 + [STOP_SYMBOL]
        token4 = [START_SYMBOL]*3 + token0 +  [STOP_SYMBOL]

        for unigram in token1:
            unigram_c[unigram] += 1

        for bigram in nltk.bigrams(token2):
            bigram_c[bigram] += 1

        for trigram in nltk.trigrams(token3):
            trigram_c[trigram] += 1
         
        for quadgram in nltk.ngrams(token4,4):
            quadgram_c[quadgram] +=1
            
    
    unigram_total = sum(unigram_c.itervalues())
    unigram_p = {unigram: math.log(unigram_c[unigram],2) - math.log(unigram_total,2)  for unigram in unigram_c}    
    
     #P(wi, wi-1) = count(wi,wi-1)/count(wi)    
    unigram_c[START_SYMBOL] = len(training_corpus)    
    bigram_p = {(x,y): math.log(bigram_c[(x,y)],2) - math.log(unigram_c[x],2)  for x,y in bigram_c}    
    
    #P(wi, wi-1, wi-2) = count(wi, wi-1, wi-2) / count(wi-1, wi-2)
    bigram_c[(START_SYMBOL,START_SYMBOL)]   = len(training_corpus)
    trigram_p = {(x,y,z): math.log(trigram_c[(x,y,z)],2) - math.log(bigram_c[(x,y)],2) for x,y,z in trigram_c }
    
    #P(wi,wi-1,wi-2,wi-3)
    trigram_c[(START_SYMBOL,START_SYMBOL,START_SYMBOL)] =len(training_corpus)
    quadgram_p = {(p,q,r,s): math.log(quadgram_c[(p,q,r,s)],2) - math.log(trigram_c[(p,q,r)],2) for p,q,r,s in quadgram_c }
    
    
    return unigram_p, bigram_p, trigram_p, quadgram_p

def extract_collocations(bigram_p, unigram_p,n):
    colloc_info = defaultdict(float)
    
    for x,y in bigram_p:
        if x <> START_SYMBOL and y <> START_SYMBOL:
           colloc_info[(x,y)] = math.log(2**bigram_p[(x,y)],2) - math.log(2**unigram_p[x],2) - math.log(2**unigram_p[y],2)
    sorted_tuple = sorted( (key for key in colloc_info), key=colloc_info.get, reverse=True)  
    
    try:
        return sorted_tuple[:n]
        
    except:
        
        return "The number supplied is greater than total number of bigrams"
            
DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
infile = open('/home/sidvash/coursera_NLP/Assignment2/data/Brown_train.txt', 'r')
corpus = infile.readlines()
infile.close()

    # calculate ngram probabilities 
unigrams, bigrams, trigrams, quadgrams = calc_probabilities(corpus)   
    
extract_collocations(bigrams, unigrams, 40)    