# -* encoding: utf-8 *-

import keras
import json as js
import pandas as pd

# in_csv = pandas.read_csv('in.csv', delimiter=' ', header=None)
in_csv_f = open('test.csv', 'r')
in_csv = []

line = in_csv_f.readline()

f = open('model.json', 'r')
json = js.load(f)
model = keras.models.model_from_json(json)
model.load_weights('weights.h5')
f.close()
# 辞書の作成
dictf = open('dict.json', 'r')
dict_str = js.load(dictf)
dictf.close()
# print(dict_str)
inv_dict = {v:k for k, v in dict_str.items()}

# TIME_STEPの読み込み
setting_f = open('settings.json', 'r')
json = js.load(setting_f)
TIME_STEP = json["TIME_STEP"]
setting_f.close()


while line:
    tmp = line.split(' ')

    word_int = []
    for i in tmp:
        if i != '\n':
            word_int.append(inv_dict[i])
    in_csv.append(word_int)
    line = in_csv_f.readline()

import numpy as np

for i in range(200):
    input_data = in_csv[i]
    # print(input_data)
    output = ""
    for x in input_data[0:min(len(input_data),TIME_STEP)]:
        output += dict_str[x]
    output += ":"
    for j in range(20):
        first = j
        # print(input_data[first:first+4])
        idata = []
        if len(input_data[first:first+TIME_STEP]) < TIME_STEP:
            for x in range(TIME_STEP - len(input_data[first:first+TIME_STEP])):
                input_data.append(0)
        idata.append(input_data[first:first + TIME_STEP])
        idata = np.array(idata)
        # print(idata.shape)
        idata = np.reshape(idata, (idata.shape[0], idata.shape[1], 1))
        # 結果の取得
        dt = model.predict(idata)
        # print(dt.shape)
        # print(dt)
        # print("max index: ", np.argmax(dt))
        output += dict_str[str(np.argmax(dt))]
    print(output)
# inv_dict = {v:k for k, v in dict_str.items()}
# print(inv_dict)
#print(dict_str)

