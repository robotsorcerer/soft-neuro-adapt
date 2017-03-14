import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.cuda
import numpy as np

from qpth.qp import QPFunction

import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, nFeatures, nCls, nHidden, nineq=12, neq=0, eps=1e-4, 
                noutputs=3,numLayers=1):
        super(LSTMModel, self).__init__()
        
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        self.cost = nn.MSELoss(size_average=False)
        self.noutputs = noutputs
        # self.neunet = nn.Sequential()
        self.lstm1  = nn.LSTM(nHidden[0],nHidden[0],num_layers=numLayers)
        self.lstm2  = nn.LSTM(nHidden[0],nHidden[1],num_layers=numLayers)
        self.lstm3  = nn.LSTM(nHidden[1],nHidden[2],num_layers=numLayers)
        self.drop   = nn.Dropout(0.3)
        self.fc1 = nn.Linear(nHidden[2], noutputs)
        self.fc2 = nn.Linear(noutputs, noutputs)

        self.M = Variable(torch.tril(torch.ones(nCls, nCls)))
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls)))
        self.G = Parameter(torch.Tensor(nineq/2, nCls).uniform_(-1,1))

        """
        define constraints, z_i, and slack variables, s_i,
        for six valves. z_i and c_i are learnable parameters
        """
        self.z0 = Parameter(torch.zeros(nCls))
        self.s0 = Parameter(torch.ones(nineq/2))
        self.z0p = Parameter(torch.zeros(nCls))
        self.s0p = Parameter(torch.ones(nineq/2))

    def forward(self, x):
        nBatch = x.size(0)
        print('x: ', x.size(0))
        # FC-ReLU-QP-FC-Softmax
        # LSTM-dropout-LSTM-dropout-lstm-dropout-FC-QP-FC
        x = x.view(nBatch, -1)
        x = self.drop(self.lstm1(x))
        x = self.drop(self.lstm2(x))
        x = self.drop(self.lstm3(x))
        x = self.fc1(x)
        """
        Q = self.cost
        #define inequality constraints for the six valves upperbounded by 1
        h0 = self.G.mv(self.z0)+self.s0-Parameter(torch.ones(nineq/2))
        #define inequality constraints for the six valves lowerbounded by 0
        h1 = self.G.mv(self.z0p)-self.s0p
        #concat h0 and h1 into h
        h = torch.cat((h0, h1),0)
        #concat G to the global G
        G = torch.cat((G, G), 0)

        e   = Variable(torch.Tensor())
        x = QPFunction(verbose=False)(x,Q,G,h,e,e)
        """        
        x = self.fc2(x)

        return F.sigmoid(x) #squash signals between 0 and 1

    # define model
    def lstm_model():
        cost          = nn.MSELoss()
        hidden        = [9, 6, 6]
        inputSize     = hidden[0]
        numLayers     = 1
        noutputs      = 3
        neunet        = nn.Sequential(
                            nn.LSTM(inputSize,hidden[0],num_layers=numLayers),
                            nn.Dropout(0.3),
                            nn.LSTM(hidden[0],hidden[1],num_layers=numLayers),
                            nn.Dropout(0.3),
                            nn.LSTM(hidden[1],hidden[2],num_layers=numLayers),
                            nn.Dropout(0.3),
                            nn.Linear(hidden[2], noutputs)
            )
        print cost
        print(neunet)

    def test_RNN_cell():
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for module in (nn.RNNCell, nn.GRUCell):
            for bias in (True, False):
                input = Variable(torch.randn(3, 10))
                hx = Variable(torch.randn(3, 20))
                cell = module(10, 20, bias=bias)
                for i in range(6):
                    hx = cell(input, hx)

                hx.sum().backward()

    def test_LSTM_cell():
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for bias in (True, False):
            input = Variable(torch.randn(3, 10))
            hx = Variable(torch.randn(3, 20))
            cx = Variable(torch.randn(3, 20))
            lstm = nn.LSTMCell(10, 20, bias=bias)
            for i in range(6):
                hx, cx = lstm(input, (hx, cx))

            (hx + cx).sum().backward()

# rnn = nn.LSTM(10,20,2)
# input = Variable(torch.randn(5,3, 10))
# h0 = Variable(torch.randn(2,3, 20))
# c0 = Variable(torch.randn(2,3,20))
# output, hn = rnn(input, (h0, c0))

# print ('output', output)
# print('hn ', hn)

# baseLSTM = LSTMModel(6,3,True)
# baseLSTM.lstm_model()

'''
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size+hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x, h):
        inp = torch.cat((x,h), 1)
        hidden = self.tanh(self.i2h(inp))
        output = self.h2o(inp)
        return hidden, output
    
    
    def get_output(self, X):
        time_steps = X.size(0)
        batch_size = X.size(1)
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        outputs = []
        hiddens = []
        for t in range(time_steps):
            hidden, output = self.forward(X[t], hidden)
            outputs.append(output)
            hiddens.append(hidden)
        return torch.cat(hiddens, 1), torch.cat(outputs, 1)
    
## Helper functions

def get_variable_from_np(X):
    return Variable(torch.from_numpy(X)).float()


def get_training_data(X, y, max_steps=10):
    inputs = []
    targets = []
    time_steps = X.shape[0]
    for i in range(0, time_steps, max_steps):
        inputs.append(get_variable_from_np(
            X[i:i+max_steps, np.newaxis, np.newaxis]))
        targets.append(get_variable_from_np(
            y[i:i+max_steps, np.newaxis, np.newaxis]))
    return torch.cat(inputs, 1), torch.cat(targets, 1)
'''