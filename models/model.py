import torch
import torch.nn as nn
import numpy
strokes = numpy.load('../data/strokes.npy')
stroke = strokes[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.LSTM()
        self.

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
