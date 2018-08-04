# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:38:51 2018

@author: user
"""
from nltk.text import TextCollection

corpus = TextCollection(['this  is sentence one',
                         'this is sentence two',
                         'this is sentence three'])
print(corpus.tf_idf('this','this is sentence four'))

