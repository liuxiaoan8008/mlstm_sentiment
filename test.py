#!usr/bin/env python
# -*- coding:utf-8 -*-

from encoder import Model

model = Model('/var/data/mlstm/model/model.npy')

print model.transform(['恩恩好好表现哦'])

