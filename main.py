# -* encoding: utf-8 *-

import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
from keras.layers.core import Dropout, Activation, Flatten, Dense
import json

'''亜流'''
def make_train_data_with_phrase(filename):
    clm_name = []
    MAX_CLM = 1000
    for i in range(MAX_CLM):
        clm_name.append(i)

    print(clm_name)

    df = pd.read_csv(filename, delimiter=' ', header=None, names=clm_name)
    df = df.fillna('-1')
    print('****** input data is *******')
    print(df)
    print('****************************')

    vocab_dict = {"-1":-500000}
    dict_index = 0

    train_x = []
    train_y = []

    for i, row in df.iterrows():
        print(i)
        print(row)
        x = []

        for word in row:
            if not word in vocab_dict: # キーを持っているか調べる
                vocab_dict[word] = dict_index # len(vocab_dict)
                print('key added :', word, ', value :', dict_index)
                dict_index += 1
            x.append(vocab_dict[word])
            train_x.append(x[:])

        '''
        for index in range(len(train_x)-1):
            train_y.append(train_x[index+1])
        '''

        print("********************")
        print("result:")
        print(train_y)
        print("*******************")

    return (vocab_dict, train_x, train_y)


# wordベースでtrainと正解データを作成する
def make_train_data(filename):
    clm_name = []
    MAX_CLM = 1200
    for i in range( MAX_CLM):
        clm_name.append(i)

    # print(clm_name)

    df = pd.read_csv(filename, delimiter=' ', header=None, names=clm_name)
    df = df.fillna('-1')
    # print('****** input data is *******')
    # print(df)
    # print('****************************')

    vocab_dict = {"-1": 0}
    vocab_str_dict = {0: "-1"}
    dict_index = 0

    train_x = []
    train_y = []

    for i, row in df.iterrows():
        print( i, ": ", row)
        tmp_train_x = []
        for word in row:
            if word not in vocab_dict: # キーを持っているか調べる
                vocab_dict[word] = len(vocab_dict)
                vocab_str_dict[vocab_dict[word]] = word # reverseしてkey: val => word のディクショナリも作成する
                print('key added :', word, ', value :', dict_index)
                dict_index += 1

            if vocab_dict[word] != 0:
                tmp_train_x.append(vocab_dict[word])
                train_x.append(vocab_dict[word])
            else:
                break
        for index in range(len(tmp_train_x)-1):
            train_y.append(tmp_train_x[index+1])
            # print("x: ", vocab_str_dict[tmp_train_x[index]], ", y: ", vocab_str_dict[train_y[len(train_y)-1]])

    #  print("********************")
    #  print("result:")
    #  print(train_y)
    #  print("*******************")
    print("****creating train_y*****")
    print(len(train_y))
    ty = []
    for i in train_y:
        print("now: ",i)
        tmp = np.zeros(len(vocab_dict))
        tmp[train_y] = 1
        ty.append(tmp)
    return train_x, ty, vocab_dict, vocab_str_dict

def lstm(data, dict):
    print("lstm func")

def generate_train_xy(data, vocab, time_step=2):
    tx, ty = [], []
    for i in range(len(data)-time_step):
        tx.append(np.array(data[i:i+time_step]))
        tmp_y = np.zeros(len(vocab))
        tmp_y[data[i+time_step]] = 1
        #ty.append(data[i+time_step])
        ty.append(tmp_y)
    return tx, ty
train_x, train_y, vocab, vocab_reverse = make_train_data('test.csv')

print(train_x)
BATCH_SIZE = 80
TIME_STEPS = 4
hidden_neurons = 1024 #1024
input_dim = 1
output_neurons = len(vocab)

train_x, train_y = generate_train_xy(train_x, vocab, TIME_STEPS)
train_x = np.array(train_x)
train_y = np.array(train_y)
# print(train_x)
print(train_x.shape)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
# e(train_x, (train_x.shape[0], train_x.shape[1], 1))

EPOCH = 600

model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, TIME_STEPS, input_dim), return_sequences=False))
#model.add(Dense(output_neurons))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4096))
model.add(Dropout(0.1))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=BATCH_SIZE,
          nb_epoch=EPOCH, verbose=1)

json_dict = model.to_json()
output_f = open('model.json', 'w')
json.dump(json_dict, output_f)

print(vocab_reverse)
dic_file = open('dict.json', 'w')
json.dump(vocab_reverse,dic_file)

model.save_weights('weights.h5')

from tensorflow.contrib.keras.python.keras import backend as K
K.clear_session()
