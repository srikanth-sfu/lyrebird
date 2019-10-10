import spacy
import sys
import argparse
import numpy as np
import os
import  pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='train',help='vocabulary build mode (for word2vec)')

args = parser.parse_args()
with open('../data/%s.txt'%(args.data)) as f:
    text_data = f.readlines()

def save_dict():
    #vocab built using these chars and a special "none of these" character.
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."? '
    chars += "'"
    vocab = {}
    for charid, char in enumerate(chars):
        vocab[char] = charid
    n_chars = len(vocab)
    out_data = []
    for data in text_data:
        data_embed = []
        data = data.replace('\n', '')
        for c in data:
            if c in chars:
                data_embed += [vocab[c]]
            else:
                #c not in vocab
                data_embed += [n_chars]
        data_embed = np.array(data_embed).astype('int32')
        out_data += [data_embed]
    np.save('../data/%s_vocab.npy'%(args.data), out_data)


if __name__ == '__main__':
    save_dict()
