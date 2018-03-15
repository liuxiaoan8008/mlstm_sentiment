#!usr/bin/env python
# -*- coding:utf-8 -*-

from encoder import Model

model = Model('/var/data/mlstm/model/model.npy')

def load_sst(path):
    data = pd.read_csv(path, encoding='utf-8')
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y

X,Y = load_sst('./data/test.csv')
Xt = model.transform(X)
for x in Xt:
	print x


