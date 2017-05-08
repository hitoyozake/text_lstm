# -* encoding: utf-8 *-

import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
from keras.layers.core import Dropout, Activation, Flatten, Dense
import json

def make_dictionary(df, na_valuestr="-1"):
    vocab_dict = {na_valuestr: 0}
    vocab_str_dict = {0: na_valuestr}

    dict_index = 1

    # 辞書を作成する
    for i, row in df.iterrows():
        print(i, ": ", row)
        tmp_train_x = []
        for word in row:
            if word not in vocab_dict: # キーを持っているか調べる
                vocab_dict[word] = len(vocab_dict)
                vocab_str_dict[vocab_dict[word]] = word # reverseしてkey: val => word のディクショナリも作成する
                print('key added :', word, ', value :', dict_index)
                dict_index += 1

            if vocab_dict[word] != 0:
                tmp_train_x.append(vocab_dict[word])
            else:
                break

    return vocab_dict, vocab_str_dict


def add_char_to_dictionary(target_str, vocab_dict):

    dict_index = len(vocab_dict)

    vocab_reverse_dict = {}

    # 辞書を作成する
    for c in target_str:
            if c not in vocab_dict: # キーを持っているか調べる
                vocab_dict[c] = len(vocab_dict)
                vocab_reverse_dict[vocab_dict[c]] = c#reverseしてkey: val => word のディクショナリも作成する
                print('key added :', c, ', value :', dict_index)
                dict_index += 1

            # if vocab_dict[c] != 0:
            #    tmp_train_x.append(vocab_dict[word])
            #    train_x.append(vocab_dict[word])
            # else:
            #    break

    return vocab_dict, vocab_reverse_dict


def make_char_dictionary(strlist):

    dict_index = 1
    vocab_dict = {}
    for s in strlist:
        vocab_dict, rev_dict = add_char_to_dictionary(s, vocab_dict)

    return vocab_dict, rev_dict



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

    # 辞書を作成する
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


def generate_xy(lines, vocab, time_step=2, train_rate = 0.8):

    x, y = [], []
    for line in lines:
        for index in range(0, len(line)-time_step):# char in lines
            substr = line[index:index+time_step]

            x_intlist = [0] * len(substr)
            for i, c in enumerate(substr):
                x_intlist[i] = vocab[c]

            y_char = line[index+time_step]
            print(y_char)
            y_buf = np.zeros(len(vocab))
            y_buf[vocab[y_char]] = 1
            x.append(x_intlist)
            y.append(y_buf)

    train_x, train_y = x[:int(len(x)*train_rate)], y[:int(len(x)*train_rate)]
    test_x, test_y = x[int(len(x)*train_rate):], y[int(len(x)*train_rate):]

    return train_x, train_y, test_x, test_y


def generate_train_xy(data, vocab, time_step=2):
    tx, ty = [], []
    for i in range(len(data)-time_step):
        tx.append(np.array(data[i:i+time_step]))
        tmp_y = np.zeros(len(vocab))
        tmp_y[data[i+time_step]] = 1
        # ty.append(data[i+time_step])
        ty.append(tmp_y)
    return tx, ty


def char_lstm(filename):
    BATCH_SIZE = 80
    TIME_STEPS = 6
    hidden_neurons = 256 #1024
    input_dim = 1
    EPOCH=80

    inputfile = open(filename, mode='r')


    strlist = []
    readlines = inputfile.readlines()

    for line in readlines:
        strlist.append(line)

    vocab, reverse_dict = make_char_dictionary(strlist)

    # train_x, train_y, test_x, test_yの作成
    train_x, train_y, test_x, test_y = generate_xy(strlist, vocab, time_step=TIME_STEPS, train_rate=0.7)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    '''
    print("***** generated trainx, trainy, testx, testy ******")

    print("train_x: ", train_x)
    print("train_y: ", train_y)
    print("test_x: ", test_x)
    print("test_y: ", test_y)

    print("*******************")
    '''

    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, TIME_STEPS, input_dim), return_sequences=False))
    # model.add(Dense(output_neurons))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
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


def string_lstm(csvfilename):
    train_x, train_y, vocab, vocab_reverse = make_train_data(csvfilename)

    print(train_x)
    BATCH_SIZE = 80
    TIME_STEPS = 4
    EPOCH = 600

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


    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, TIME_STEPS, input_dim), return_sequences=False))
    # model.add(Dense(output_neurons))
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
    json.dump(vocab_reverse, dic_file)

    model.save_weights('weights.h5')
    from keras.backend.tensorflow_backend import clear_session
    clear_session()
    # from tensorflow import reset_default_graph
    # reset_default_graph()


if __name__ == '__main__':
    char_lstm("test.txt")


