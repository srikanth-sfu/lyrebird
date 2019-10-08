import torch
import torch.nn as nn
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_prefix', default='./models/',help='model checkpoint directory')
parser.add_argument('-c', '--conditional', action='store_true',help='conditional v/s unconditionali')

args = parser.parse_args()


strokes = numpy.load('../data/strokes.npy')
with open('../data/sentences.txt') as f:
    texts = f.readlines()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if not args.c:
            self.lstm1 = nn.LSTM(3,256)
        else:
            raise
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,3)
        #TODO Incorporate mixture density model

    def forward(x=None):
        pass

model = Net().cuda()

def load(mode='train'):
    pass

def train():
    model.train()
    

def save():
    pass

def test():
    model.test()

    pass


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
