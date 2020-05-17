# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-


import os.path, pickle
import numpy as np
from ProcessData import get_data
from Evaluate import evaluation_NER
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.Model_CNN_CRF import CNN_CRF_char
from NNstruc.Model_CNN_CRF import CNN_CRF_char_posi
from NNstruc.Model_CNN_CRF import CNN_CRF_char_posi_attention_5
from NNstruc.Model_CNN_CRF import attenCNN_3_CRF_char_posi_attention_5


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

        senty = test_y[0][si]
        ttag = []

        for word in senty:
            next_index = np.argmax(word)
            if next_index == 0:
                break
            next_token = index2word[next_index]
            ttag.append(next_token)

        result = []
        result.append(ptag)
        result.append(ttag)
        testresult.append(result)

    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult)

    return P, R, F, PR_count, P_count, TR_count


def SelectModel(modelname, charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                char_W, posi_W, word_W,
                input_seq_lenth,
                char_k, posi_k, word_k, batch_size):

    nn_model = None

    if modelname is 'CNN_CRF_char_posi':
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

    elif modelname is 'CNN_CRF_char_posi_attention_5':
        nn_model = CNN_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              wordvobsize=wordvobsize,
                                              char_W=char_W, posi_W=posi_W, word_W=word_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, word_k=word_k, batch_size=batch_size)

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

    # modelname = 'CNN_CRF_char'
    # modelname = 'CNN_CRF_char_posi'
    # modelname = 'CNN_CRF_char_posi_attention_5'
    modelname = 'attenCNN_3_CRF_char_posi_attention_5'

    print(modelname)
    resultdir = "./data/result/"

    # ------------------1
    trainfile = './data/weiboNER/weiboNER_2nd_conll.train.dev.BIOES.txt'
    testfile = './data/weiboNER/weiboNER_2nd_conll.test.BIOES.txt'
    # char2v_file = "./data/preEmbedding/weiboNER_Char2Vec.txt"
    char2v_file = "./data/preEmbedding/weiboNER_Char2Vec_pure.txt"
    word2v_file = "./data/preEmbedding/sgns.weibo.bigram-char"
    dataname = 'weiboNER.data_WSUI.3'
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

    inputs_train_x = [trainx_char, trainx_posi]
    inputs_train_y = [trainy]
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
            infer_e2e_model(nnmodel, modelfile, inputs_test_x, inputs_test_y, idex_2target, resultdir,
                            batch_size=batch_size, testfile=testfile)

