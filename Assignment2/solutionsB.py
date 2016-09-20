import sys
import nltk
import math
import time
import re
from collections import defaultdict
import itertools

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):

    
    brown_tags = map(lambda x: re.sub(r"([^\s]+)\/([^\s]+)", r"\2", x), brown_train)
    brown_tags = map(lambda x: [START_SYMBOL]*2+ x.strip().split() + [STOP_SYMBOL], brown_tags)

    brown_words = map(lambda x: re.sub(r"([^\s]+)\/([^\s]+)", r"\1", x), brown_train)
    brown_words = map(lambda x: [START_SYMBOL]*2+ x.strip().split() + [STOP_SYMBOL], brown_words)

    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    
    bigram_c = defaultdict(int)
    trigram_c = defaultdict(int)
    
    brown_tags_bi = map(lambda x: x[1:], brown_tags)
        
    bigram_tuples = map(lambda x: list(nltk.bigrams(x)), brown_tags_bi)
    trigram_tuples = map(lambda x: list(nltk.trigrams(x)), brown_tags)
    
    for sublist in bigram_tuples:
        for bigram in sublist:
            bigram_c[bigram] += 1
    for sublist in trigram_tuples:
        for trigram in sublist:
            trigram_c[trigram] += 1
     
    bigram_c[(START_SYMBOL,START_SYMBOL)] = len(brown_tags)
    q_values = {(x,y,z): math.log(trigram_c[(x,y,z)],2) - math.log(bigram_c[(x,y)],2)   for x,y,z in trigram_c }           
    
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    
    known_words =set()
    
    unigram_c = defaultdict(int)
    
    for sublist in brown_words:
        for unigram in sublist:
            unigram_c[(unigram)] += 1
    
    for key,value in unigram_c.iteritems():
        if value > RARE_WORD_MAX_FREQ :
           known_words.add(key)
           
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    
    for i, sentence in enumerate(brown_words):
        for j, word in enumerate(sentence):
            if word not in known_words:
                brown_words[i][j] = RARE_SYMBOL
                
    return brown_words

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    #P(word|tag) = c(word,tag) / C(tag) 
    word_tag_count = defaultdict(int) 
    tag_count = defaultdict(int)
    e_value = {}
    
    for sent_word,sent_tag in zip(brown_words_rare, brown_tags):
        for item in zip(sent_word,sent_tag):
            word_tag_count[item] +=1
    
    for sentence in brown_tags:
        for tag in sentence:
            tag_count[tag] +=1
         
    e_value = { (word,tag):math.log(word_tag_count[(word,tag)],2) - math.log(tag_count[tag],2)    for word,tag in word_tag_count }  

    
    return  e_value,set(tag_count)  

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
     
    #initialization
    tagged = []
    tag_list = list(taglist)     
    tag_tuples = []
    
    pi = defaultdict(float)
    bp = {}
    
    pi[(0,START_SYMBOL, START_SYMBOL)] = 1
    

    #Creating all combinations of (u,v) tag tuples where u <> '*' and v <> '*' 
    for i in range(len(tag_list)):
        for j in range(len(tag_list)):
            if tag_list[i] == START_SYMBOL or tag_list[j] == START_SYMBOL:
               continue
            tag_tuples.append((tag_list[i],tag_list[j]))
            
    #initial condition
    for u,v in tag_tuples:
        pi[(0,u,v)] = 0
    
    
    #Viterbi:
    for tokens_orig in brown_dev_words:
        tokens = [token if token in known_words else RARE_SYMBOL for token in tokens_orig]
        for k in range(1,len(tokens)+1):
            for u in taglist:
                for v in taglist:
                    max_score = float('-Inf')
                    for w in tag_list:
                        score = pi[(k-1,w,u)] + q_values.get((w,u,v),LOG_PROB_OF_ZERO)  + e_values.get((tokens[k-1],v),LOG_PROB_OF_ZERO)
                        if score > max_score:
                           max_score = score 
                           max_tag = w    
                    pi[(k,u,v)] = max_score 
                    bp[(k,u,v)] = max_tag
        
        #Calculating last term, q(stop|yn-1, yn)
        max_score = float('-Inf')
        tags =[]
        
        for (u,v) in tag_tuples:
            score = pi[k-1,w,u] + q_values.get((u,v,STOP_SYMBOL),LOG_PROB_OF_ZERO)  
            if score > max_score:
               tags = [v,u]
       
        for count,k in enumerate(range(len(tokens_orig)-2,0,-1)):
            tags.append(bp[(k+2,tags[count+1],tags[count])])
        #reversing tags    
        tags.reverse()
         
        #creating tagged sentence        
        tagged_sent = []
         
        for k in range(len(tokens)):
            tagged_sent += [tokens_orig[k], "/", tags[k], " "]
        tagged_sent.append('\n')    
        
        tagged.append("".join(tagged_sent))   
             
    return tagged
    

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i], brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for sentence in brown_dev_words:
        tagged.append(' '.join([word + '/' + tag for word, tag in trigram_tagger.tag(sentence)]) + '\n')
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
