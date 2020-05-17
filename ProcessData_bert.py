# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

__author__ = 'JIA'
import numpy as np
import pickle, jieba, codecs
import json
import re
import math
from keras_bert import Tokenizer


config_path = './data/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './data/bert/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
id2token_dict = {}

with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
        id2token_dict[token_dict[token]] = token

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


def get_Character_index(files):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    max_s = 0
    tarcount=0
    count = 1
    num = 0
    token = 0

    for file in files:
        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                if num > max_s:
                    max_s = num
                # print(max_s, '  ', num)
                num = 0
                continue
            token += 1

            num += 1
            sourc = line.strip('\r\n').rstrip('\n').split('\t')
            # print(sourc)
            if not source_vob.__contains__(sourc[0]):
                source_vob[sourc[0]] = count
                sourc_idex_word[count] = sourc[0]
                count += 1

            if not target_vob.__contains__(sourc[len(sourc)-1]):
                target_vob[sourc[len(sourc)-1]] = tarcount
                target_idex_word[tarcount] = sourc[len(sourc)-1]
                tarcount += 1

        f.close()

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def make_idx_Char_index(file, max_s, target_vob, istest=False):

    data_s_all = []
    data_t_all = []

    f = codecs.open(file, 'r', encoding='utf-8')
    fr = f.readlines()
    count = 0
    data_t = []
    data_s = ''

    for line in fr:
        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            # for inum in range(0, num):
            #     data_s.append(0)
            #
            #     targetvec = np.zeros(len(target_vob) + 1)
            #     targetvec[0] = 1
            #     data_t.append(targetvec)
            #     # data_t.append(0)
            # print(data_s)
            # print(data_t)
            # if istest:
            #     padding = np.zeros(len(target_vob))
            #     # padding[0] = 1
            #     if num > 0:
            #         data_t = data_t + np.tile(padding, (num+1, 1)).tolist()

            data_s_all.append(data_s)

            data_t_all.append(data_t)
            data_t = []
            data_s = ''
            count = 0
            continue

        sent = line.strip('\r\n').rstrip('\n').split('\t')
        data_s += sent[0]

        targetvec = np.zeros(len(target_vob))
        targetvec[target_vob[sent[len(sent)-1]]] = 1
        data_t.append(targetvec)
        # data_t.append(target_vob[sent[1]])
        count += 1

    f.close()

    return data_s_all, data_t_all


def seq_padding(X, max_s, padding=0):
    # L = [len(x) for x in X]
    # ML = max(L)
    ML = max_s
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def seq_padding2(X, max_s, target_vob):

    padding = np.zeros(len(target_vob))
    padding[target_vob['O']] = 1

    return np.array([
        x + np.tile(padding, (max_s - len(x), 1)).tolist() if len(x) < max_s else x for x in X
    ])


class data_generator:
    def __init__(self, data, data_posi, label, batch_size=32, maxlen=50, target_vob=None):
        self.data = data
        self.data_posi = data_posi
        self.label = label
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.target_vob = target_vob
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps
    def __iter__(self):

        while True:
            idxs = np.arange(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, Y, PI = [], [], [], []
            for i in idxs:

                d = self.data[i]
                text = d[:self.maxlen]
                x1, x2 = tokenizer.encode(first=text, max_len=self.maxlen+2)
                y = self.label[i]
                pi = self.data_posi[i]

                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                PI.append(pi)



                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1, self.maxlen+2)
                    X2 = seq_padding(X2, self.maxlen+2)
                    Y = seq_padding2(Y, self.maxlen, self.target_vob)
                    nPI = np.asarray(PI, dtype="int32")
                    # for tti, tt in enumerate(x1):
                    #     print(tti, id2token_dict[tt])
                    #     print(y[tti])

                    # print(X1.shape, X2.shape, Y.shape)

                    yield ([X1, X2, nPI], Y)
                    [X1, X2, Y, PI] = [], [], [], []


class data_generator_4test:
    def __init__(self, data, data_posi, batch_size=32, maxlen=50):
        self.data = data
        self.data_posi = data_posi
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data))

            X1, X2, PI = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[:self.maxlen]
                x1, x2 = tokenizer.encode(first=text, max_len=self.maxlen+2)
                X1.append(x1)
                X2.append(x2)
                PI.append(self.data_posi[i])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1, self.maxlen+2)
                    X2 = seq_padding(X2, self.maxlen+2)
                    nPI = np.asarray(PI, dtype="int32")
                    yield ([X1, X2, nPI])
                    [X1, X2, PI] = [], [], []



def get_data(trainfile, testfile, datafile, datafile0, batch_size=8, maxlen = 50):

    print('loading data of posi ...')
    _, _, _, _, \
    train_posi_all, test_posi, _, _, \
    _, _, posi_vob, _, \
    _, _, \
    _, posi_W, _, \
    _, posi_k, _, _ = pickle.load(open(datafile0, 'rb'))


    char_vob, idex_2char, target_vob, idex_2target, max_s = get_Character_index({trainfile, testfile})

    print("source char size: ", char_vob.__len__())
    print("max_s: ", max_s)
    print("source char: ", len(idex_2char))
    print("target vocab size: ", len(target_vob), str(target_vob))
    print("target vocab size: ", len(idex_2target))

    alldata, alldata_label = make_idx_Char_index(trainfile, max_s, target_vob)
    test, test_label = make_idx_Char_index(testfile, max_s, target_vob, istest=True)
    print('train len  ', alldata.__len__(), len(alldata_label))
    print('test len  ', test.__len__(), len(test_label), len(test_label[0]))

    # 按照9:1的比例划分训练集和验证集
    random_order = np.arange(len(alldata))
    np.random.shuffle(random_order)
    train_data = [alldata[j] for i, j in enumerate(random_order) if i % 5 != 0]
    train_label = [alldata_label[j] for i, j in enumerate(random_order) if i % 5 != 0]
    dev_data = [alldata[j] for i, j in enumerate(random_order) if i % 5 == 0]
    dev_label = [alldata_label[j] for i, j in enumerate(random_order) if i % 5 == 0]
    train_posi = [train_posi_all[j] for i, j in enumerate(random_order) if i % 5 != 0]
    dev_posi = [train_posi_all[j] for i, j in enumerate(random_order) if i % 5 == 0]
    print(len(train_data), len(train_posi), len(train_label), len(dev_data), len(dev_posi), len(dev_label))

    # return train_data, train_label, dev_data, dev_label, test, test_label, target_vob, idex_2target, max_s

    # train_D = data_generator(train, train_label, batch_size=batch_size, maxlen=max_s)
    # test_D = data_generator(test, test_label, batch_size=batch_size, maxlen=max_s)

    print("dataset created!")
    out = open(datafile, 'wb')
    pickle.dump([train_data, train_posi, train_label,
                 dev_data, dev_posi, dev_label,
                 test, test_posi, test_label,
                 target_vob, idex_2target,
                 posi_vob, posi_W, posi_k,
                 max_s], out, 0)
    out.close()
    print("dataset finished !")

