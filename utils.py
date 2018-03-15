import numpy as np
import tensorflow as tf
import pprint
import os
import argparse
import time
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from HTMLParser import HTMLParser
from sklearn.linear_model import LogisticRegression

def preprocess(text, front_pad='\n ', end_pad=' '):
    'change the next two lines because the original code is for Python 3'
    # this function removes the escape characters from the input string [text] and also
    # adds a newline to the beggining and a space and the end as start and end tokens for each review
    h = HTMLParser()
    text = unicode(text)
    text = h.unescape(text)

    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    text = text.encode('utf-8')
    text = bytearray(text) # for python 2
    return text


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


class HParams(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)



def concatenate_for_encoder(checkpoint_path):

    weights_list = np.load(checkpoint_path)

    new_list = []

    W_embedding = weights_list[0] # W_embedding.shape = (256,64)
    new_list.append(W_embedding)

    # these 4 matrices are (64,4096)
    Wix = weights_list[6]
    Wfx = weights_list[12]
    Wox = weights_list[9]
    Whx = weights_list[3]

    # in encoder.py, these matrices are concatenated and saved as 1.npy
    # wx is the variable name in the encoder.py graph
    wx = np.concatenate((Wix,Wfx,Wox,Whx), axis=1)  # wx.shape = (64,16384)
    new_list.append(wx)


    # These matrices are saved seperately in .npy files, then concatenated from within the encoder.py script
    # before being used for the wh variable in the encoder.py graph. These are all of shape (4096,4096)
    Wim = weights_list[7]
    new_list.append(Wim)

    Wfm = weights_list[13]
    new_list.append(Wfm)

    Wom = weights_list[10]
    new_list.append(Wom)

    Whm = weights_list[4]
    new_list.append(Whm)

    # Wmx is used to calculate the m vector, this is saved as 6.npy
    Wmx = weights_list[1]   # Wmx.shape = (64,4096)
    new_list.append(Wmx)

    # Wmh is used to calculate the m vector, this is saved as 7.npy
    Wmh = weights_list[2]   # Wmh.shape = (4096,4096)
    new_list.append(Wmh)

    # The bias variables are concatenated and saved in the 8.npy file
    # These are 4 (4096,) vectors used to calculate z in encoder.py
    Wib = weights_list[8]
    Wfb = weights_list[14]
    Wob = weights_list[11]
    Whb = weights_list[5]

    b = np.concatenate((Wib,Wfb,Wob,Whb), axis=1)
    # remove singleton dimension
    b = b.squeeze()
    new_list.append(b)

    # Coefficients used for weight normalizationn for the wx matrix, used in the calculation of z
    # the following vectores are conctenated and saved as 9.npy
    gix = weights_list[19]
    gfx = weights_list[23]
    gox = weights_list[21]
    ghx = weights_list[17]
    gx = np.concatenate((gix,gfx,gox,ghx))
    new_list.append(gx)

    # Coefficients used for the  weight normalization for the wh matrix used in the calculation of z,
    # these are concatenated and saved as 10.npy
    gim = weights_list[20]
    gfm = weights_list[24]
    gom = weights_list[22]
    ghm = weights_list[18]
    gh = np.concatenate((gim,gfm,gom,ghm))
    new_list.append(gh)

    # gmx and gmh are the weight normalization coefficients used for wmx and wmh in the calculation of m
    # gmx
    gmx = weights_list[15]
    new_list.append(gmx)

    # gmh
    gmh = weights_list[16]
    new_list.append(gmh)

    # These aren't used for the representation extraction but extract the softmax weights too
    Classifier_w = weights_list[25]
    new_list.append(Classifier_w)

    Classifier_b = weights_list[26]
    new_list.append(Classifier_b)

    return new_list
