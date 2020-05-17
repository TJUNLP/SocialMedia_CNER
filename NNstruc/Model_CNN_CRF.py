# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-


from keras import backend as K
from keras.layers import Flatten,Lambda, Conv2D, MaxPooling2D, Reshape
from keras.layers.core import Dropout, Activation, Permute, RepeatVector, Reshape
from keras.layers.merge import concatenate, Concatenate, multiply, Dot, add, Add
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D
from keras.layers import GlobalMaxPooling1D, RepeatVector, AveragePooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras import optimizers
from keras.layers.normalization import BatchNormalization



def CNN_CRF_char_posi(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)
    posi_embedding = Dense(25, activation=None)(posi_embedding)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding],axis=-1)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, posi_input], model)

    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi_attention_5(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):

    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(100, activation=None)(posi_embedding)
    posi_embedding_dense = Dropout(0.5)(posi_embedding_dense)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding_dense],axis=-1)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    attention = TimeDistributed(Dense(4, activation='tanh'))(cnns)
    posi_representation = TimeDistributed(Activation('softmax'))(attention)

    posi_embedding_atten = Dense(25, activation=None)(posi_representation)
    embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten],axis=-1)

    cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn4_atten = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn2_atten = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn5_atten = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding_atten)
    cnns_atten = concatenate([cnn5_atten, cnn3_atten, cnn4_atten, cnn2_atten], axis=-1)
    cnns_atten = BatchNormalization(axis=1)(cnns_atten)
    cnns_atten = Dropout(0.5)(cnns_atten)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns_atten)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, posi_input], model)

    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def attenCNN_3_CRF_char_posi_attention_5(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):

    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(100, activation=None)(posi_embedding)
    posi_embedding_dense = Dropout(0.5)(posi_embedding_dense)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding_dense],axis=-1)

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
    embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten],axis=-1)

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

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(representation)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, posi_input], model)

    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')


    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(char_embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(char_embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(char_embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(char_embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, posi_input], model)

    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def reverse_sequence(kvar):

    # kvar = K.variable(np.random.random((2, 5)))
    # kvar_zeros = K.zeros_like(kvar)
    # print(K.eval(kvar))

    kvar2 = K.reverse(kvar, axes=1)
    # print(K.eval(kvar2))
    return kvar2


def zero_softmax(x, axis=-1):

    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    th = K.round(e + 0.5 - K.epsilon())
    s = K.sum(e, axis=axis, keepdims=True) + K.epsilon()
    return e * th / s
