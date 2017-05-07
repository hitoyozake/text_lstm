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

print(dict_str)

inv_dict = {v:k for k, v in dict_str.items()}

print("*********")
print("len: ", len(dict_str), "len inv: ", len(inv_dict))

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
    for x in input_data[0:min(len(input_data),4)]:
        output += dict_str[x]
    output += ":"
    for j in range(20):
        first = j
        # print(input_data[first:first+4])
        idata = []
        if len(input_data[first:first+4]) < 4:
            for x in range(4 - len(input_data[first:first+4])):
                input_data.append(0)
        idata.append(input_data[first:first + 4])
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

