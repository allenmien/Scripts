# -*-coding:utf-8-*-
# python3.6
"""
@Time   : 2018/8/1 18:03
@Author : Mark
@File   : nltk_script.py
"""
import nltk
from nltk import FreqDist

corpus = "this is my sentence "\
            "this is my life "\
            "this is the day"
tokens = nltk.word_tokenize(corpus)
print(tokens)

fdist = FreqDist(tokens)
print(fdist["is"])

standard_freq_vector = fdist.most_common(50)
size = len(standard_freq_vector)
print(standard_freq_vector)

# look-up
def position_lookup(v):
    res = dict()
    counter = 0
    for word in v:
        res[word[0]] = counter
        counter += 1
    return res

standard_position_dict = position_lookup(standard_freq_vector)
standard_position_dict = sorted(
standard_position_dict.items(), key=lambda d: d[1], reverse=False)
print(standard_freq_vector)

sentence = "this is cool"
freq_vector = [0] * size
tokens = nltk.word_tokenize(sentence)
for word in tokens:
    try:
        freq_vector[standard_position_dict[word]] += 1
    except KeyError:
        continue
print(freq_vector)
