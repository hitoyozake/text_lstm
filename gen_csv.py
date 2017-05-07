#!/usr/bin/env python
# coding: utf-8
import MeCab
import csv
import sys


def StringCSV2WordCSV( input_filename, output_filename ):

    # -d はシステム辞書を使うという意味

    mcbTagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    fo = open(output_filename, 'w')

    for line in csv.reader(open(input_filename, 'r')):
        if len(line) == 0:
            continue
        line = line[0]
        print('Input String:', line)
        fo.write(mcbTagger.parse(line))

    fo.close()



if __name__ == '__main__':
    param = sys.argv

    if len(param) != 3:
        print( 'usage : ', param[0], ' in_strCSVFilename out_strCSVFilename')
    else:
        StringCSV2WordCSV(param[1], param[2])