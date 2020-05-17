# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import pickle, codecs, keras
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from ProcessData import get_data
from Evaluate import evaluation_NER
from Evaluate import evaluation_NER_error
from keras.layers import Flatten,Lambda,Conv2D
from keras.layers.core import Dropout, Activation, Permute, RepeatVector
from keras.layers.merge import concatenate, Concatenate, multiply, Dot
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
# from keras.losses import my_cross_entropy_withWeight
from NNstruc.Model_RNN_CRF import BiLSTM_CRF
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_un_bigramChar
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char_posi
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char_posi_word
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char_posi_word_2
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_word
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char_posi_attention
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char_posi_attention_2
from NNstruc.Model_RNN_CRF import BiLSTM_CRF_char_posi_attention_5

from NNstruc.Model_CNN_CRF import CNN_CRF_char_posi
from NNstruc.Model_CNN_CRF import CNN_CRF_char
from NNstruc.Model_CNN_CRF import CNN_CRF_char_posi_attention
from NNstruc.Model_CNN_CRF import BiCNN_CRF_char_posi_attention_5
from NNstruc.Model_CNN_CRF import IDCNN_CRF_char_posi_attention_5
from NNstruc.Model_CNN_CRF import CNN_CRF_char_posi_attention_5
from NNstruc.Model_CNN_CRF import BiCNN_CRF_char
from NNstruc.Model_CNN_CRF import attenCNN_CRF_char_posi_attention_5
from NNstruc.Model_CNN_CRF import attenCNN_2_CRF_char_posi_attention_5
from NNstruc.Model_CNN_CRF import attenCNN_3_CRF_char_posi_attention_5


def test_getmoretag(file):

    data_t_all = []
    data_t = []

    f = codecs.open(file, 'r', encoding='utf-8')
    for line in f.readlines():

        if line.__len__() <= 1:
            data_t_all.append(data_t)
            data_t = []
            continue

        sent = line.strip('\r\n').rstrip('\n').split('\t')
        data_t.append(sent[1])

    f.close()

    return data_t_all


def test_model(nn_model, inputs_test_x, test_y, index2word, resultfile ='', batch_size=10, testfile=''):

    index2word[0] = ''

    predictions = nn_model.predict(inputs_test_x)
    testresult = []
    for si in range(0, len(predictions)):
        sent = predictions[si]
        # print('predictions',sent)
        ptag = []
        for word in sent:
            next_index = np.argmax(word)
            # if next_index == 0:
            #     break
            next_token = index2word[next_index]
            ptag.append(next_token)
        # print('next_token--ptag--',str(ptag))

        senty = test_y[0][si]
        ttag = []

        for word in senty:
            next_index = np.argmax(word)
            if next_index == 0:
                break
            next_token = index2word[next_index]
            # if word > 0:
            #     if flag == 0:
            #         flag = 1
            #         count+=1
            ttag.append(next_token)
        # print(si, 'next_token--ttag--', str(ttag))
        result = []
        result.append(ptag)
        result.append(ttag)

        # if si == 34:
        #     print('ptag____', ptag[23:33])
        #     print('ttag____', ttag[23:33])

        testresult.append(result)
        # print(result.shape)
    # print('count-----------',count)
    # pickle.dump(testresult, open(resultfile, 'w'))
    #  P, R, F = evaluavtion_triple(testresult)

    # moretag_list = test_getmoretag(testfile)

    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult)
    # evaluation_NER2(testresult)
    # print (P, R, F)
    # evaluation_NER_error(testresult)

    return P, R, F, PR_count, P_count, TR_count


def atten_test_model(nn_model, inputs_test_x, index2word, resultfile='', batch_size=10):


    intermediate_layer_model = keras.models.Model(inputs=nn_model.input,
                                                  outputs=nn_model.get_layer('posi_atten').output)
    predictions = intermediate_layer_model.predict(inputs_test_x, verbose=1, batch_size=batch_size)

    for si in range(5, 6):
        wlist = {}
        seq = inputs_test_x[0][si]
        sent = predictions[si]

        for ci, cc in enumerate(seq):
            if seq[ci] == 0:
                break
            wlist[index2word[cc]] = list(sent[ci])
            print(index2word[cc], inputs_test_x[1][si].tolist())
        print(si, wlist)




def SelectModel(modelname, charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                char_W, posi_W, word_W,
                input_seq_lenth,
                char_k, posi_k, word_k, batch_size):
    nn_model = None
    if modelname is 'Model_BiLSTM_CRF':
        nn_model = BiLSTM_CRF(charvocabsize=charvocabsize,
                                    targetvocabsize=targetvocabsize,
                                    posivocabsize=posivocabsize,
                                    wordvobsize=wordvobsize,
                                    char_W=char_W, posi_W=posi_W, word_W=word_W,
                                    input_seq_lenth=input_seq_lenth,
                                    char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char':
        nn_model = BiLSTM_CRF_char(charvocabsize=charvocabsize,
                              targetvocabsize=targetvocabsize,
                              posivocabsize=posivocabsize,
                              wordvobsize=wordvobsize,
                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                              input_seq_lenth=input_seq_lenth,
                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char_posi':
        nn_model = BiLSTM_CRF_char_posi(charvocabsize=charvocabsize,
                              targetvocabsize=targetvocabsize,
                              posivocabsize=posivocabsize,
                              wordvobsize=wordvobsize,
                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                              input_seq_lenth=input_seq_lenth,
                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char_posi_word':
        nn_model = BiLSTM_CRF_char_posi_word(charvocabsize=charvocabsize,
                              targetvocabsize=targetvocabsize,
                              posivocabsize=posivocabsize,
                              wordvobsize=wordvobsize,
                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                              input_seq_lenth=input_seq_lenth,
                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char_posi_word_2':
        nn_model = BiLSTM_CRF_char_posi_word_2(charvocabsize=charvocabsize,
                              targetvocabsize=targetvocabsize,
                              posivocabsize=posivocabsize,
                              wordvobsize=wordvobsize,
                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                              input_seq_lenth=input_seq_lenth,
                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_word':
        nn_model = BiLSTM_CRF_word(charvocabsize=charvocabsize,
                              targetvocabsize=targetvocabsize,
                              posivocabsize=posivocabsize,
                              wordvobsize=wordvobsize,
                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                              input_seq_lenth=input_seq_lenth,
                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char_posi_attention':
        nn_model = BiLSTM_CRF_char_posi_attention(charvocabsize=charvocabsize,
                               targetvocabsize=targetvocabsize,
                               posivocabsize=posivocabsize,
                               wordvobsize=wordvobsize,
                               char_W=char_W, posi_W=posi_W, word_W=word_W,
                               input_seq_lenth=input_seq_lenth,
                               char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char_posi_attention_2':
        nn_model = BiLSTM_CRF_char_posi_attention_2(charvocabsize=charvocabsize,
                               targetvocabsize=targetvocabsize,
                               posivocabsize=posivocabsize,
                               wordvobsize=wordvobsize,
                               char_W=char_W, posi_W=posi_W, word_W=word_W,
                               input_seq_lenth=input_seq_lenth,
                               char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiLSTM_CRF_char_posi_attention_5':
        nn_model = BiLSTM_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                               targetvocabsize=targetvocabsize,
                               posivocabsize=posivocabsize,
                               wordvobsize=wordvobsize,
                               char_W=char_W, posi_W=posi_W, word_W=word_W,
                               input_seq_lenth=input_seq_lenth,
                               char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'CNN_CRF_char_posi':
        nn_model = CNN_CRF_char_posi(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'CNN_CRF_char':
        nn_model = CNN_CRF_char(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'CNN_CRF_char_posi_attention':
        nn_model = CNN_CRF_char_posi_attention(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiCNN_CRF_char_posi_attention_5':
        nn_model = BiCNN_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'IDCNN_CRF_char_posi_attention_5':
        nn_model = IDCNN_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'CNN_CRF_char_posi_attention_5':
        nn_model = CNN_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'BiCNN_CRF_char':
        nn_model = BiCNN_CRF_char(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'attenCNN_CRF_char_posi_attention_5':
        nn_model = attenCNN_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                  targetvocabsize=targetvocabsize,
                                  posivocabsize=posivocabsize,
                                  wordvobsize=wordvobsize,
                                  char_W=char_W, posi_W=posi_W, word_W=word_W,
                                  input_seq_lenth=input_seq_lenth,
                                  char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

    elif modelname is 'attenCNN_2_CRF_char_posi_attention_5':
        nn_model = attenCNN_2_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                                      targetvocabsize=targetvocabsize,
                                                      posivocabsize=posivocabsize,
                                                      wordvobsize=wordvobsize,
                                                      char_W=char_W, posi_W=posi_W, word_W=word_W,
                                                      input_seq_lenth=input_seq_lenth,
                                                      char_k=char_k, posi_k=posi_k, word_k=word_k,
                                                      batch_size=batch_size)

    elif modelname is 'attenCNN_3_CRF_char_posi_attention_5':
        nn_model = attenCNN_3_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                                      targetvocabsize=targetvocabsize,
                                                      posivocabsize=posivocabsize,
                                                      wordvobsize=wordvobsize,
                                                      char_W=char_W, posi_W=posi_W, word_W=word_W,
                                                      input_seq_lenth=input_seq_lenth,
                                                      char_k=char_k, posi_k=posi_k, word_k=word_k,
                                                      batch_size=batch_size)



    return nn_model


def train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y, resultdir, npoches=100, batch_size=50, retrain=False):

    if retrain:
        nn_model.load_weights(modelfile)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # val_acc val_crf_viterbi_accuracy
    checkpointer = ModelCheckpoint(filepath=modelfile+".best_model.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)
    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1
        nn_model.fit(x=inputs_train_x,
                     y=inputs_train_y,
                     batch_size=batch_size,
                     epochs=increment,
                     verbose=1,
                     shuffle=True,
                     class_weight='auto',
                     validation_split=0.2,
                     callbacks=[reduce_lr, checkpointer])

        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, inputs_test_x, inputs_test_y, idex_2target,
                                                          resultfile='',
                                                          batch_size=batch_size)
        if F > maxF:
            maxF = F
            earlystop = 0
            nn_model.save_weights(modelfile, overwrite=True)

        print(nowepoch, 'P= ', P, '  R= ', R, '  F= ', F, '>>>>>>>>>>>>>>>>>>>>>>>>>>maxF= ', maxF)

        if earlystop > 50:
            break


    return nn_model


def infer_e2e_model(nn_model, modelfile, inputs_test_x, inputs_test_y, idex_2target, resultdir,
                    batch_size=50, testfile=''):

    nn_model.load_weights(modelfile)
    resultfile = resultdir + "result-" + 'infer_test'

    loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=0, batch_size=batch_size)
    print('\n test_test score:', loss, acc)

    P, R, F, PR_count, P_count, TR_count = test_model(nn_model, inputs_test_x, inputs_test_y, idex_2target, resultfile,
                                                      batch_size, testfile)
    print('P= ', P, '  R= ', R, '  F= ', F)

    if os.path.exists(modelfile+".best_model.h5"):
        print('test best_model ......>>>>>>>>>>>>>>> ' + modelfile+".best_model.h5" )
        nn_model.load_weights(modelfile+".best_model.h5")
        resultfile = resultdir + "best_model.result-" + 'infer_test'
        loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=0, batch_size=batch_size)
        print('\n test_test best_model score:', loss, acc)

        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, inputs_test_x, inputs_test_y, idex_2target,
                                                          resultfile,
                                                          batch_size)
        print('best_model ... P= ', P, '  R= ', R, '  F= ', F)



if __name__ == "__main__":
    '''
    weiboNER-->
    the number of entity in test is 418
    the number of entity in train and dev is 2283
    '''

    modelname = 'Model_BiLSTM_CRF'
    modelname = 'BiLSTM_CRF_char'
    modelname = 'BiLSTM_CRF_char_posi'
    # modelname = 'BiLSTM_CRF_char_posi_word'
    # modelname = 'BiLSTM_CRF_char_posi_word_2'
    # modelname = 'BiLSTM_CRF_word'
    modelname = 'BiLSTM_CRF_char_posi_attention'
    modelname = 'BiLSTM_CRF_char_posi_attention_2'
    modelname = 'BiLSTM_CRF_char_posi_attention_5'

    # modelname = 'CNN_CRF_char_posi'
    # modelname = 'CNN_CRF_char'

    modelname = 'CNN_CRF_char_posi_attention_5'
    # modelname = 'IDCNN_CRF_char_posi_attention_5'
    modelname = 'BiCNN_CRF_char_posi_attention_5'
    # modelname = 'BiCNN_CRF_char'
    modelname = 'attenCNN_CRF_char_posi_attention_5'
    modelname = 'attenCNN_2_CRF_char_posi_attention_5'
    modelname = 'attenCNN_3_CRF_char_posi_attention_5'

    print(modelname)
    resultdir = "./data/result/"

    # ------------------1
    trainfile = './data/weiboNER/weiboNER_2nd_conll.train.dev.BIOES.txt'
    testfile = './data/weiboNER/weiboNER_2nd_conll.test.BIOES.txt'
    # char2v_file = "./data/preEmbedding/weiboNER_Char2Vec.txt"
    char2v_file = "./data/preEmbedding/weiboNER_Char2Vec_pure.txt"
    word2v_file = "./data/preEmbedding/sgns.weibo.bigram-char"
    dataname = 'weiboNER.data_WSUI.3'#!!!!!!!!!!!!!.3.pure
    # dataname = 'weiboNER.data_WSUI.2'
    datafile = "./data/model_data/" + dataname + ".pkl"
    batch_size = 8
    # ------------------1

    # # ------------------2
    # trainfile = './data/MSRA/train.txt.BIOES.txt'
    # testfile = './data/MSRA/test.txt.BIOES.txt'
    # char2v_file = "./data/preEmbedding/MSRA_sogounews_Char2Vec.txt"
    # word2v_file = ""
    # dataname = 'MSRA.data_WSUI.2'
    # datafile = "./data/model_data/" + dataname + ".pkl"
    # batch_size = 32
    # # ------------------2


    retrain = False
    Test = True
    valid = False
    Label = True
    if not os.path.exists(datafile):
        print("Process data....")
        get_data(trainfile=trainfile, testfile=testfile,
                 w2v_file='', c2v_file=char2v_file,
                 datafile=datafile, w2v_k=300, c2v_k=100, maxlen=50)

    print("data has extisted: " + datafile)
    print('loading data ...')
    train, train_label, test, test_label,\
    train_posi, test_posi, train_word, test_word,\
    char_vob, target_vob, posi_vob, word_vob,\
    idex_2char, idex_2target,\
    character_W, posi_W, word_W,\
    character_k, posi_k, word_k, max_s = pickle.load(open(datafile, 'rb'))

    trainx_char = np.asarray(train, dtype="int32")
    trainy = np.asarray(train_label, dtype="int32")
    trainx_posi = np.asarray(train_posi, dtype="int32")
    trainx_word = np.asarray(train_word, dtype="int32")
    testx_char = np.asarray(test, dtype="int32")
    testy = np.asarray(test_label, dtype="int32")
    testx_posi = np.asarray(test_posi, dtype="int32")
    testx_word = np.asarray(test_word, dtype="int32")

    # inputs_train_x = [trainx_char, trainx_posi, trainx_word]
    inputs_train_x = [trainx_char, trainx_posi]
    inputs_train_y = [trainy]
    # inputs_test_x = [testx_char, testx_posi, testx_word]
    inputs_test_x = [testx_char, testx_posi]
    inputs_test_y = [testy]

    for inum in range(3, 8):

        nnmodel = None
        nnmodel = SelectModel(modelname,
                              charvocabsize=len(char_vob),
                              targetvocabsize=len(target_vob),
                              posivocabsize=len(posi_vob),
                              wordvobsize=len(word_vob),
                              char_W=character_W,
                              posi_W=posi_W,
                              word_W=word_W,
                              input_seq_lenth=max_s,
                              char_k=character_k, posi_k=posi_k, word_k=word_k,
                              batch_size=batch_size)


        modelfile = "./data/model/" + dataname + '__' + modelname + "_" + str(inum) + ".h5"
        # modelfile = "./data/model/" + dataname + '__' + modelname + "_posi_word_" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Training model....")
            print(modelfile)
            nnmodel.summary()
            train_e2e_model(nnmodel, modelfile, inputs_train_x, inputs_train_y, resultdir,
                            npoches=120, batch_size=batch_size, retrain=False)
        else:
            if retrain:
                print("ReTraining model....")
                train_e2e_model(nnmodel, modelfile, inputs_train_x, inputs_train_y, resultdir,
                            npoches=120, batch_size=batch_size, retrain=retrain)

        if Test:
            print("test model....")
            print(modelfile)
            # nnmodel.summary()
            infer_e2e_model(nnmodel, modelfile, inputs_test_x, inputs_test_y, idex_2target, resultdir,
                            batch_size=batch_size, testfile=testfile)

            # atten_test_model(nnmodel, inputs_test_x, idex_2char, resultdir, batch_size)



# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES="" python Model.py