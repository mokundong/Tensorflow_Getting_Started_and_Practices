# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 09:23:56 2018

@author: mkd
"""

import numpy as np
import tensorflow as tf
import random
import collections
import jieba
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.manifold import TSNE

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.family'] = 'STsong'
mpl.rcParams['font.size'] = 20

training_file = '人体阴阳与电能.txt'
#中文字
def get_ch_label(txt_file):
    labels = ""
    with open(txt_file,'rb') as f:
        for label in f:
            labels = labels + label.decode('gb2312')
    return labels

#分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    #使用空格将字符串分开
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci,[-1,])
    return training_ci

def build_dataset(words,n_words):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))#返回top (n_words - 1)列表
    #print(count)
    dictionary = dict()
    for word , _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count #'UNK'计数
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


training_data = get_ch_label(training_file)
#print("总字数",len(training_data))
training_ci = fenci(training_data)
#print("总字数",len(training_ci))
training_label, count, dictionary, words = build_dataset(training_ci, 350)
#word_size = len(dictionary)
#print("字典词数",word_size)
#print('Sample data',training_label[:10],[words[i] for i in training_label[:10]])
#print(training_label)
#########获取批次数据##########
data_index = 0
def generate_batch(data,batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window + 1 #每一个样本由前skip_window + 当前target + 后skip_window组成
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window #target在buffer中的索引为skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]      
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) -span) % len(data)
    return batch,labels

batch, labels = generate_batch(training_label,batch_size=8,num_skips=2,skip_window=1)

for i in range(8):#先循环8次，将组合好的样本与标签打印出来
    print(batch[i],words[batch[i]],'->',labels[1,0],words[labels[i,0]])
    


