import torch
import torch.nn as nn
import numpy
import argparse
import random

strokes = numpy.load('../data/strokes.npy')
with open('../data/sentences.txt') as f:
    texts = f.readlines()

def slice_save(idxs, name='train'):
    t = [texts[x] for x in idxs]
    s = [strokes[x] for x in idxs]
    numpy.save('../data/%s.npy'%(name),s)
    with open('../data/%s.txt'%(name), 'w') as f:
        f.write('\n'.join(t))
    f.close()

n_data = len(texts)
n_train = n_data/2
n_val = n_data/4
rem = range(n_data)
train = random.sample(rem, n_train)
rem = list(set(rem) - set(train))
val = random.sample(rem, n_val)
test = list(set(rem) - set(val))

slice_save(train,'train')
slice_save(val,'val')
slice_save(test,'test')
