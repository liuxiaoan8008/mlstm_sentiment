#!usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from encoder import Model

base_dir = '/var/data/mlstm/'

model = Model(base_dir + 'models/model.npy')

def load_sst(path):
    data = pd.read_csv(path, encoding='utf-8')
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


trX,trY = load_sst('./data/train.csv')
teX,teY = load_sst('./data/test.csv')
print trX[0]
print trY[0]

if not os.path.exists(base_dir + 'features'):
    os.makedirs(base_dir + 'features')   
    trXt = model.transform(trX)
    teXt = model.transform(teX)

    np.save(base_dir + 'features/trXt',trXt)
    np.save(base_dir + 'features/teXt',trXt)

else:
	trXt = np.load(base_dir + 'features/trXt.npy')
    teXt = np.load(base_dir + 'features/teXt.npy')



