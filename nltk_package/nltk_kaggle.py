#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time   : 2018/8/3
@Author : Mark
@File   : nltk_kaggle.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

df_train = pd.read_csv('./input/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('./input/test.csv', encoding='ISO-8859-1')

df_desc = pd.read_csv('./input/product_descriptions.csv')

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
print(df_all.shape)

df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

stemmer = SnowballStemmer('english')

df_train2 = df_all.loc[df_train.index]

# s中的单词去除噪音
def str_stemmer(s):
    return ' '.join([stemmer.stem(word) for word in s.lower().split()])


# str1中的单词在str2中出现的次数
def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:
                                                                  str_stemmer(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x:
                                                   len(x.split())).astypc(np.int64)
df_all['commons_in_title'] = df_all.apply(lambda x:
                                          str_common_word
                                          (x['search_term'],
                                           x['product_title']),
                                          axis=1)
df_all['commons_in_desc'] = df_all.apply(lambda x:
                                         str_common_word
                                         (x['search_term'],
                                          x['product_description']),
                                         axis=1)
df_all = df_all.drop(['search_term', 'product_title', 'product_description'],
                     axis=1)

df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]
 
test_ids = df_test['id']
y_train = df_train['relevance'].values


