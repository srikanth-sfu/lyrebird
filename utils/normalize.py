import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='train',help='vocabulary build mode (for word2vec)')

args = parser.parse_args()

train_data = np.load('../data/train.npy')
filename = '../data/%s.npy'%(args.data)

target_data = np.load(filename)

x = np.concatenate([x[:,1] for x in train_data])
y = np.concatenate([y[:,2] for y in train_data])

mean_offset = (x.mean(), y.mean())
var_offset = (x.std(), y.std())
for dataid in range(len(target_data)):
    target_data[dataid][:,1] -= mean_offset[0]
    target_data[dataid][:,2] -= mean_offset[1]
    target_data[dataid][:,1] /= var_offset[0]
    target_data[dataid][:,2] /= var_offset[1]
np.save(filename,target_data)
