import math
import nltk
import time
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    #Empty count dicts    
    unigram_c = defaultdict(int)
    bigram_c = defaultdict(int)
    trigram_c = defaultdict(int)
    
    for sentence in training_corpus:
        token0 = sentence.strip().split()
        token1 = token0 + [STOP_SYMBOL]
        token2 = [START_SYMBOL] + token0 + [STOP_SYMBOL]
        token3 = [START_SYMBOL] +[START_SYMBOL]+ token0 + [STOP_SYMBOL]

        for unigram in token1:
            unigram_c[unigram] += 1

        for bigram in nltk.bigrams(token2):
            bigram_c[bigram] += 1

        for trigram in nltk.trigrams(token3):
            trigram_c[trigram] += 1
    
    
    unigram_total = sum(unigram_c.itervalues())
    unigram_p = {unigram: math.log(unigram_c[unigram],2) - math.log(unigram_total,2)  for unigram in unigram_c}    
    
     #P(wi, wi-1) = count(wi,wi-1)/count(wi)    
    unigram_c[START_SYMBOL] = len(training_corpus)    
    bigram_p = {(x,y): math.log(bigram_c[(x,y)],2) - math.log(unigram_c[x],2)  for x,y in bigram_c}    
    
    #P(wi, wi-1, wi-2) = count(wi, wi-1, wi-2) / count(wi-1, wi-2)
    bigram_c[(START_SYMBOL,START_SYMBOL)]   = len(training_corpus)
    trigram_p = {(x,y,z): math.log(trigram_c[(x,y,z)],2) - math.log(bigram_c[(x,y)],2) for x,y,z in trigram_c }
    
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    
    if n==1:    
       for sentence in corpus:
           token0 = sentence.strip().split()
           tokens =  token0 + [STOP_SYMBOL]
           prob_sent = 0
           if all(ngrams in ngram_p for ngrams in tokens):  #if n gram doesn't exist return log_prob = -1000
              for x in tokens:
                  prob_sent += ngram_p[x]
              scores.append(prob_sent)
              
           else:
              scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)        
        
    elif n==2:    
         for sentence in corpus:
             token0 = sentence.strip().split()
             tokens = list(nltk.bigrams([START_SYMBOL] + token0 + [STOP_SYMBOL]))
             prob_sent = 0
             if all(ngrams in ngram_p for ngrams in tokens):
                for x in tokens:
                    prob_sent += ngram_p[x]
                scores.append(prob_sent)
                 
             else:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)  
             
    elif n==3:    
         for sentence in corpus:
             token0 = sentence.strip().split()
             tokens = list(nltk.trigrams([START_SYMBOL] +[START_SYMBOL]+ token0 + [STOP_SYMBOL]))
             prob_sent = 0
             if all(ngrams in ngram_p for ngrams in tokens):  #if n gram doesn't exist return log_prob = -1000
                for x in tokens:
                    prob_sent += ngram_p[x]
                scores.append(prob_sent)  
             
             else:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)  
             
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lambda_ = 1/3.0
    for sentence in corpus:
        token0 = sentence.strip().split()
        token3 = list(nltk.trigrams([START_SYMBOL] +[START_SYMBOL]+ token0 + [STOP_SYMBOL]))
        interpolated_score = 0
        for trigram in token3:
            try:
                p3 = trigrams[trigram]
                p2 = bigrams[trigram[1:3]]
                p1 = unigrams[trigram[2]]
                interpolated_score += math.log(lambda_*(2**p3+2**p2+2**p1) ,2)
            
            except KeyError:
                interpolated_score = MINUS_INFINITY_SENTENCE_LOG_PROB
                
        scores.append(interpolated_score)
    
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question okens_new2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
