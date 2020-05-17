#! -*- coding:utf-8 -*-

import os, pickle
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from ProcessData_bert import get_data, data_generator, data_generator_4test
from Evaluate import evaluation_NER
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

config_path = './data/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './data/bert/chinese_L-12_H-768_A-12/vocab.txt'


def Model_Bert(targetvocabsize):

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    posi_input = Input(shape=(max_s,), dtype='int32')

    x = bert_model([x1_in, x2_in])
    bert_out = Lambda(lambda x: x[:, 1:max_s+1])(x)

    char_embedding = Dense(100, activation=None)(bert_out)

    TimeD = TimeDistributed(Dense(targetvocabsize))(char_embedding)


    outp = Activation('softmax')(TimeD)

    model = Model([x1_in, x2_in, posi_input], outp)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5), # 用足够小的学习率
        metrics=['accuracy']
    )

    return model


def Model_Bert_attenCNN_3_CRF_char_posi_attention_5(targetvocabsize, posivocabsize, posi_W,
                     input_seq_lenth, posi_k, batch_size=16):

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(input_seq_lenth+2,))
    x2_in = Input(shape=(input_seq_lenth+2,))

    bert_out = bert_model([x1_in, x2_in])
    bert_out = Lambda(lambda x: x[:, 1:max_s + 1])(bert_out)
    char_embedding = Dense(100, activation=None)(bert_out)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(100, activation=None)(posi_embedding)
    posi_embedding_dense = Dropout(0.5)(posi_embedding_dense)

    embedding = concatenate([char_embedding, posi_embedding_dense],axis=-1)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    attention = TimeDistributed(Dense(4, activation='tanh'))(cnns)
    posi_representation = TimeDistributed(Activation('softmax'), name='posi_atten')(attention)

    posi_embedding_atten = Dense(25, activation=None)(posi_representation)
    embedding_atten = concatenate([char_embedding, posi_embedding_atten],axis=-1)

    cnn1_atten = Conv1D(100, 1, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn2_atten = Conv1D(100, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn4_atten = Conv1D(100, 4, activation='relu', strides=1, padding='same')(embedding_atten)

    cnn1_atten = TimeDistributed(RepeatVector(1))(cnn1_atten)
    cnn2_atten = TimeDistributed(RepeatVector(1))(cnn2_atten)
    cnn3_atten = TimeDistributed(RepeatVector(1))(cnn3_atten)
    cnn4_atten = TimeDistributed(RepeatVector(1))(cnn4_atten)

    embedding_atten_re = Lambda(reverse_sequence)(embedding_atten)
    cnn2_atten_re = Conv1D(100, 2, activation='relu', strides=1, padding='same')(embedding_atten_re)
    cnn2_atten_re = Lambda(reverse_sequence)(cnn2_atten_re)
    cnn3_atten_re = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten_re)
    cnn3_atten_re = Lambda(reverse_sequence)(cnn3_atten_re)
    cnn4_atten_re = Conv1D(100, 4, activation='relu', strides=1, padding='same')(embedding_atten_re)
    cnn4_atten_re = Lambda(reverse_sequence)(cnn4_atten_re)

    cnn2_atten_re = TimeDistributed(RepeatVector(1))(cnn2_atten_re)
    cnn3_atten_re = TimeDistributed(RepeatVector(1))(cnn3_atten_re)
    cnn4_atten_re = TimeDistributed(RepeatVector(1))(cnn4_atten_re)

    pinjie = Lambda(lambda x: K.concatenate([x[0], x[1], x[2], x[3], x[4], x[5], x[6]], axis=2))\
        ([cnn1_atten, cnn2_atten, cnn3_atten, cnn4_atten,
          cnn2_atten_re, cnn3_atten_re, cnn4_atten_re])

    quary = posi_embedding_atten
    quary = TimeDistributed(RepeatVector(7))(quary)
    values = Dropout(0.5)(pinjie)
    score = concatenate([quary, values], axis=-1)
    score = TimeDistributed(Dense(1, activation='tanh'))(score)
    score = TimeDistributed(Flatten())(score)
    score = TimeDistributed(Activation('softmax'))(score)
    score = TimeDistributed(Reshape((7, 1)))(score)
    representation = Lambda(lambda x: x[0] * x[1])([pinjie, score])
    representation = TimeDistributed(Flatten())(representation)
    representation = BatchNormalization(axis=1)(representation)
    representation = Dropout(0.5)(representation)

    TimeD = TimeDistributed(Dense(targetvocabsize))(representation)

    # outp = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize, sparse_target=False)
    outp = crflayer(TimeD)

    Models = Model([x1_in, x2_in, posi_input], outp)
    # Models.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=Adam(1e-5), # 用足够小的学习率
    #     metrics=['accuracy']
    # )
    Models.compile(loss=crflayer.loss_function,
                   optimizer=Adam(lr=1e-5),
                   metrics=[crflayer.accuracy])

    return Models


def reverse_sequence(kvar):

    kvar2 = K.reverse(kvar, axes=1)
    return kvar2


def test_model(nn_model, index2word, batch_size=10):

    # index2word[0] = ''
    test_D = data_generator_4test(test, test_posi, batch_size=batch_size, maxlen=max_s)

    predictions = nn_model.predict_generator(generator=test_D.__iter__(), steps=len(test_D), verbose=1)
    print(len(test), len(predictions))

    testresult = []
    for si in range(0, len(predictions)):
        sent = predictions[si]

        ptag = []
        senty = test_label[si]
        ttag = []

        for wi, word in enumerate(senty):
            next_index = np.argmax(word)

            next_token = index2word[next_index]
            ttag.append(next_token)

            next_index = np.argmax(sent[wi])

            next_token = index2word[next_index]
            ptag.append(next_token)

        result = []
        result.append(ptag)
        result.append(ttag)

        testresult.append(result)

    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult)

    return P, R, F, PR_count, P_count, TR_count


def train_e2e_model(nn_model, modelfile, npoches=100, batch_size=50):

    nn_model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, min_lr=1e-7)

    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1

        train_D = data_generator(train_data, train_posi, train_label, batch_size=batch_size, maxlen=max_s, target_vob=target_vob)
        valid_D = data_generator(dev_data, dev_posi, dev_label, batch_size=batch_size, maxlen=max_s, target_vob=target_vob)

        print('data_generator finished.......')


        nn_model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=1,
            class_weight='auto',
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            shuffle=True,
            # callbacks=[reduce_lr]
        )

        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, idex_2target, batch_size=batch_size)
        if F > maxF:
            maxF = F
            earlystop = 0
            nn_model.save_weights(modelfile, overwrite=True)

        print(nowepoch, 'P= ', P, '  R= ', R, '  F= ', F, '>>>>>>>>>>>>>>>>>>>>>>>>>>maxF= ', maxF)

        if earlystop > 20:
            break


    return nn_model


def infer_e2e_model(nn_model, modelfile, idex_2target, batch_size=50):

    nn_model.load_weights(modelfile)

    P, R, F, PR_count, P_count, TR_count = test_model(nn_model, idex_2target, batch_size=batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)



if __name__ == "__main__":


    # modelname = 'Model_Bert'
    # modelname = 'Model_Bert_CNN_CRF'
    modelname = 'Model_Bert_attenCNN_3_CRF_char_posi_attention_5'

    print(modelname)
    resultdir = "./data/result/"

    trainfile = './data/weiboNER/weiboNER_2nd_conll.train.dev.BIOES.txt'
    testfile = './data/weiboNER/weiboNER_2nd_conll.test.BIOES.txt'
    dataname = 'weibo.bert.auto.posi.'
    datafile = "./data/model_data/" + dataname + ".pkl"

    dataname0 = 'weiboNER.data_WSUI.3'
    datafile0 = "./data/model_data/" + dataname0 + ".pkl"

    batch_size = 8


    Test = True
    valid = False
    Label = True

    if not os.path.exists(datafile):
        print("Process data....")

        get_data(trainfile=trainfile, testfile=testfile,
                 datafile=datafile, datafile0=datafile0,
                 batch_size=batch_size, maxlen=50)

    print("data has extisted: " + datafile)


    for inum in range(111, 114):

        print('loading data ...')

        train_data, train_posi, train_label,\
        dev_data, dev_posi, dev_label,\
        test, test_posi, test_label,\
        target_vob, idex_2target,\
        posi_vob, posi_W, posi_k, max_s = pickle.load(open(datafile, 'rb'))


        nnmodel = None

        if modelname == 'Model_Bert':
            nnmodel = Model_Bert(targetvocabsize=len(target_vob))

        elif modelname == 'Model_Bert_attenCNN_3_CRF_char_posi_attention_5':
            nnmodel = Model_Bert_attenCNN_3_CRF_char_posi_attention_5(targetvocabsize=len(target_vob),
                                                                      posivocabsize=len(posi_vob), posi_W=posi_W,
                                                                      input_seq_lenth=max_s, posi_k=posi_k,
                                                                      batch_size=batch_size)

        modelfile = "./data/model/" + dataname + 'BERT__' + modelname + "_" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Training model....")
            print(modelfile)
            train_e2e_model(nnmodel, modelfile, npoches=100, batch_size=batch_size)


        if Test:
            print("test model....")
            print(modelfile)
            infer_e2e_model(nnmodel, modelfile, idex_2target, batch_size=batch_size)

        del nnmodel
        import gc
        gc.collect()
