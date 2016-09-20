# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:46:24 2016

@author: sidvash
"""
import math
import nltk
import time
from collections import defaultdict
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

""" Basic algo check
def create_text(corpus_dict):
    essay = []
    prev_word = '*'
    next_word = ''    
    
    while(next_word <> 'STOP'):
        next_word = max((y for x,y in corpus_dict if x==prev_word), key=corpus_dict.get)
        essay.append(next_word)
        prev_word = next_word
        print next_word
        
        
    return essay
"""

def create_tri_text(corpus_dict, bistring):
    trigram_c = defaultdict(int)
    
    tokens = bistring.strip().split()
    first_bigram = (tokens[0], tokens[1])
    essay = []
    prev_bi_tuple = first_bigram
    next_word = ''    
    essay.append(first_bigram[0])
    essay.append(first_bigram[1])
    
    while(next_word <> 'STOP'):
        try:
            next_tri_tuple = max(((x,y,z) for x,y,z in corpus_dict if (x,y)==prev_bi_tuple), key=corpus_dict.get)
            trigram_c[next_tri_tuple] +=1        
        except:
            return 'Sorry!No such bi-gram found in the corpus'
            
        if trigram_c[next_tri_tuple] > 1:
           sorted_tuple = sorted(((x,y,z) for x,y,z in corpus_dict if (x,y)==prev_bi_tuple), key=corpus_dict.get, reverse=True) 
           rand_num = random.randint(1,len(sorted_tuple)-1)           
           next_tri_tuple = sorted_tuple[rand_num]  
           next_word = next_tri_tuple[2]  
           essay.append(next_word)
           
        else:
           next_word =   next_tri_tuple[2]      
           essay.append(next_word)
           
        prev_bi_tuple = (next_tri_tuple[1], next_tri_tuple[2] ) 
        
       
    
    return_essay = ' '.join(essay) 
    
    return return_essay    


def create_quad_text(corpus_dict, string):
    quadgram_c = defaultdict(int)
    tokens = string.strip().split()
    essay = []
    if len(tokens) <1:
        return 'Please input a string with at least one word'
        
    if len(tokens) == 1:
        tokens = [START_SYMBOL]*2 + tokens
        first_trigram = (tokens[0], tokens[1], tokens[2])
        essay.append(tokens[2])
        
    if len(tokens) == 2:    
        tokens = [START_SYMBOL] + tokens
        first_trigram = (tokens[0], tokens[1], tokens[2])
        essay.append(tokens[1])
        essay.append(tokens[2])
        
    else:
        first_trigram = (tokens[0], tokens[1], tokens[2])
        essay.append(tokens[0])
        essay.append(tokens[1])
        essay.append(tokens[2])
        
        
    prev_tri_tuple = first_trigram
    next_word = ''    

    
    while(next_word <> 'STOP'):
        try:
            next_quad_tuple = max(((p,q,r,s) for p,q,r,s in corpus_dict if (p,q,r)==prev_tri_tuple), key=corpus_dict.get)
            quadgram_c[next_quad_tuple] +=1        
        except:
            return 'Sorry!No such tri-gram string found in the corpus'
            
        if quadgram_c[next_quad_tuple] > 1:
           sorted_tuple = sorted( ( (p,q,r,s) for p,q,r,s in corpus_dict if (p,q,r)==prev_tri_tuple), key=corpus_dict.get, reverse=True) 
           rand_num = random.randint(1,len(sorted_tuple)-1)           
           next_quad_tuple = sorted_tuple[rand_num]  
           next_word = next_quad_tuple[3]  
           essay.append(next_word)
           
        else:
           next_word =   next_quad_tuple[3]      
           essay.append(next_word)
           
        prev_tri_tuple = (next_quad_tuple[1], next_quad_tuple[2],next_quad_tuple[3]) 
        
       
    
    return_essay = ' '.join(essay) 
    
    return return_essay    



DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
infile = open('/home/sidvash/coursera_NLP/Assignment2/data/Brown_train.txt', 'r')
corpus = infile.readlines()
infile.close()

    # calculate ngram probabilities 
unigrams, bigrams, trigrams, quadgrams = calc_probabilities(corpus)

    
    
    


 