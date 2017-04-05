# coding=utf-8
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.cuda
import numpy as np
import numpy.random as npr

from qpth.qp import QPFunction

import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    '''
    nn.LSTM Parameters:
        input_size  – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        num_layers  – Number of recurrent layers.

    Inputs: input, (h_0, c_0)
        input (seq_len, batch, input_size)
        h_0 (num_layers * num_directions, batch, hidden_size)
        c_0 (num_layers * num_directions, batch, hidden_size)

    Outputs: output, (h_n, c_n)
        output (seq_len, batch, hidden_size * num_directions)
        h_n (num_layers * num_directions, batch, hidden_size)
        c_n (num_layers * num_directions, batch, hidden_size):

    QP Layer:
        nz = 6, neq = 0, nineq = 12, QPenalty = 0.1
    '''
    def __init__(self, nz, neq, nineq, Qpenalty, inputSize, nHidden, batchSize, noutputs=3, numLayers=2):

        super(LSTMModel, self).__init__()

        # QP Parameters
        nx = nz * 2  #cause inequality is double sided (see my notes)
        self.neq = neq
        self.nineq = nineq
        self.nz = nz
        self.nHidden = nHidden

        print('nHidden: ', nHidden)

        self.Q = Variable(Qpenalty*torch.eye(nx).double())
        # self.L = Parameter(torch.potrf(Q))        
        G = torch.eye(nx).double()
        for i in range(nz):
            G[i][i] *= -1
        self.G = Variable(G)
        self.h = Variable(torch.ones(nx).double())
        self.A = Parameter(torch.rand(nx, nx).double())
        self.b = Variable(torch.ones(self.A.size(0)).double())

        def qp_layer(x):
            nBatch = x.size(0)
            Q = self.Q.unsqueeze(0).expand(nBatch, nx, nx)
            G = self.G.unsqueeze(0).expand(nBatch, nx, nx)
            h = self.h.unsqueeze(0).expand(nBatch, self.nineq)
            A = self.A.unsqueeze(0).expand(nBatch, nx, nx)
            b = self.b.unsqueeze(0).expand(nBatch, nx)
            e = Variable(torch.Tensor())
            x = QPFunction()(x.double(), Q, G, h, e, e).float()
            return x
        self.qp_layer = qp_layer

        # Backprop Through Time (Recurrent Layer) Params
        self.cost = nn.MSELoss(size_average=False)
        self.noutputs = noutputs
        self.num_layers = numLayers
        self.inputSize = inputSize
        self.nHidden = nHidden
        self.batchSize = batchSize
        self.noutputs = noutputs
        self.criterion = nn.MSELoss(size_average=False)
        
        #define recurrent and linear layers
        self.lstm1  = nn.LSTM(inputSize,nHidden[0],num_layers=numLayers)
        self.lstm2  = nn.LSTM(nHidden[0],nHidden[1],num_layers=numLayers)
        self.lstm3  = nn.LSTM(nHidden[1],nHidden[2],num_layers=numLayers)
        self.fc     = nn.Linear(nHidden[2], noutputs)
        self.drop   = nn.Dropout(0.3)

    def forward(self, x):
        nBatch = x.size(0)

        # Set initial states 
        h0 = Variable(torch.Tensor(self.num_layers, self.batchSize, self.nHidden[0])) 
        c0 = Variable(torch.Tensor(self.num_layers, self.batchSize, self.nHidden[0]))        
        # Forward propagate RNN layer 1
        out, _ = self.lstm1(x, (h0, c0)) 
        out = self.drop(out)
        
        # Set hidden layer 2 states 
        h1 = Variable(torch.Tensor(self.num_layers, self.batchSize, self.nHidden[1])) 
        c1 = Variable(torch.Tensor(self.num_layers, self.batchSize, self.nHidden[1]))        
        # Forward propagate RNN layer 2
        out, _ = self.lstm2(out, (h1, c1))  
        
        # Set hidden layer 3 states 
        h2 = Variable(torch.Tensor(self.num_layers, self.batchSize, self.nHidden[2])) 
        c2 = Variable(torch.Tensor(self.num_layers, self.batchSize, self.nHidden[2]))        
        # Forward propagate RNN layer 2
        out, _ = self.lstm3(out, (h2, c2)) 
        out = self.drop(out) 
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :]) 

        #Now add QP Layer
        out = out.view(nBatch, -1) 
        return self.qp_layer(out)

class OptNet(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""
    def __init__(self, n, nz, neq, nineq, Qpenalty):
        super().__init__()
        '''
        nz = 6, n = 12, neq = 0, nineq = 12, QPenalty = 0.1
        '''
        nx = n * 2  #cause inequality is double sided (see my notes)
        self.Q = Variable(Qpenalty*torch.eye(nx).double())
        self.G = Variable(-torch.eye(nx).double())
        self.h = Variable(torch.zeros(nx).double())
        self.A = 1000*npr.randn(neq, nz)
        self.b = Variable(torch.zeros(self.A.size(0)).double())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -puzzles.view(nBatch,-1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))

        return QPFunction(verbose=False)(p.double(), Q, G, h, A, b).float().view_as(puzzles)


class unusedFunctions():

    # def __init__():

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
        print(cost)
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