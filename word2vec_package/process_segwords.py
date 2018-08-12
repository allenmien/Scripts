# -*-coding:utf-8-*-
# @Time    : 18/8/8 09:14
# @Author  : Mark
# @File    : process_segwords.py

import jieba
import sys

reload(sys)
sys.setdefaultencoding('utf8')
count = 0

f2 = open("wiki.zh.text.seg", 'a')
with open("wiki.zh.text") as f1:
    for line in f1.xreadlines():
        line.replace('\t', '').replace('\n', '').replace(' ', '')
        seg_list = jieba.cut(line, cut_all=False)
        f2.write(" ".join(seg_list))
        count += 1
        if count % 1000 == 0:
            print(str(count))

f1.close()
f2.close()
