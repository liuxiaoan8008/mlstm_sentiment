#! /usr/bin/env python
# encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def build_data(filename,out_file1,out_file2):
    positive_f = open(out_file1,'w')
    negative_f = open(out_file2,'w')

    with open(filename) as f:
        for line in f:
            line = unicode(line.strip(),'utf-8')
            line_s = line.split(',')
            if line_s[0] == u'negative':
                negative_f.write(line_s[1].replace(u' ','')+'\n')
            elif line_s[0] ==u'positive':
                positive_f.write(line_s[1].replace(u' ','')+'\n')

    positive_f.close()
    negative_f.close()


def build_data_new(filename,out_file):
    out_f = open(out_file,'w')
    out_f.write('label,sentence\n')

    with open(filename) as f:
        for line in f:
            line = unicode(line.strip(),'utf-8')
            line_s = line.split(',')
            if line_s[0] == u'negative':
                out_f.write('0,'+line_s[1].replace(u' ','')+'\n')
            elif line_s[0] ==u'positive':
                out_f.write('1,'+line_s[1].replace(u' ','')+'\n')


build_data_new('./data/sentiment_XS_test.txt','./1/test.csv')

# build_data('/Users/liuxiaoan/Downloads/sentiment_XS_test.txt','positive_data_test.txt','negative_data_test.txt')