import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.utils.data as utils
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_prefix', default='./models/',help='model checkpoint directory')
parser.add_argument('-c', '--conditional', action='store_true',help='conditional v/s unconditionali')
parser.add_argument('-s', '--seed', default=1, type=int,help='random seed')

args = parser.parse_args()

def load_mode(mode='train'):
    strokes = np.load('../data/%s.npy'%(mode))
    strokes = [torch.Tensor(x).cuda() for x in strokes]
    if args.conditional:
        text = np.load('../data/%s_vocab.npy'%(mode))
        text = [torch.Tensor(x).cuda() for x in text]
        return strokes, ds
    else:
        return strokes

def load(mode='train'):
    return load_mode(mode)

torch.manual_seed(args.seed)

train_dataloader = load('train')
test_dataloader = load('test')
val_dataloader = load('val')

class UnconditionalNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(3,400,num_layers=1)
        self.lstm2 = nn.LSTM(400,400,num_layers=1)
        self.lstm3 = nn.LSTM(400,121,num_layers=1)
        self.pi = nn.Softmax(dim=2)
        self.gaussian_const = np.sqrt(2*np.pi)
        self.gaussian_const = 1/self.gaussian_const
        self.mu = nn.Linear(2,40)
        self.sigma = nn.Linear(2,40)


    def forward(x,prev1,prev2,prev3):
        o1, (h1,c1) = self.lstm1(x,prev1)
        x2 = torch.cat([h1,x], dim=2)
        
        o2, (h2,c2) = self.lstm2(x2,prev2)
        x3 = torch.cat([h1,h2], dim=2)

        o, (h3,c3) = self.lstm3(x3,prev3)
        
        e = o.narrow(2,0,1)
        e = F.sigmoid(-e)
        
        mu1,mu2,pi,sigma1,sigma2,rho = o.narrow(2,1,121).chunk(6,dim=2)
        
        pi = self.pi(pi)
        sigma1 = torch.exp(sigma1)
        sigma2 = torch.exp(sigma2)
        rho = F.tanh(rho)
        
        return (h1,c1), (h2,c2), e,pi, mu1, mu2, sigma1, sigma2, rho, sigma

class ConditionalNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(3,400,num_layers=1)
        self.lstm2 = nn.LSTM(400, 400, num_layers=1)
        self.lstm3 = nn.LSTM(300,121, num_layers=1)
        
    def forward(x=None):
        pass

def train_unconditional():
    model.train()

def save_ckpt(model):
    if args.conditional:
        filename = '../model_ckpt/model_conditional.pth'
    else:
        filename = '../model_ckpt/model_unconditional.pth'
        torch.save()

def test_unconditional(dl):
    model.eval()
    pass

def test_conditional(dl):
    model.eval()
    pass

def train_unconditional():
    model.train()


if not args.conditional:
    model = UnconditionalNet().cuda()
else:
    model = ConditionalNet().cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.0001, eps=0.0001,momentum=0.9, alpha=0.95)

if args.conditional:
    train = train_conditional
    test = test_conditional
else:
    train = train_unconditional
    test = test_conditional

for i in range(100):
    train()
    test(test_dataloader)
    
