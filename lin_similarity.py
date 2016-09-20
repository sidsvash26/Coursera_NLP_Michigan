# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:19:49 2016

@author: sidvash
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')

dog.lin_similarity(cat, brown_ic)


################### Porter Stemmer
from nltk.stem import *
stemmer = PorterStemmer()
stemmer.stem('computational')