# /usr/bin/env python2
# coding=utf-8
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from qpth.qp import QPFunction
torch.set_default_tensor_type('torch.DoubleTensor')

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
    def __init__(self, args, inputSize, nHidden,
                 batchSize, noutputs, numLayers):

        super(LSTMModel, self).__init__()

        # QP Parameters
        # nx = nz     #cause inequality is double sided (see my notes)
        # self.neq = neq
        # self.nineq = nineq
        # self.nz = nz
        self.args = args
        # Backprop Through Time (Recurrent Layer) Params
        self.noutputs   = noutputs
        self.num_layers = numLayers
        self.inputSize  = inputSize
        self.nHidden    = nHidden
        self.batchSize  = batchSize
        self.noutputs   = noutputs

        #define recurrent and linear layers
        self.lstm1  = nn.LSTM(inputSize, nHidden[0], num_layers=numLayers, bias=False, batch_first=False, dropout=0.3)
        self.lstm2  = nn.LSTM(nHidden[0],nHidden[1], num_layers=numLayers, bias=False, batch_first=False, dropout=0.3)
        self.lstm3  = nn.LSTM(nHidden[1],nHidden[2], num_layers=numLayers, bias=False, batch_first=False, dropout=0.3)

        if args.lastLayer=='linear':
            self.fc     = nn.Linear(nHidden[2], noutputs)
            self.criterion  = nn.MSELoss(size_average=False)
        elif args.lastLayer=='softmax':
            self.fc     = F.softmax
            self.criterion  = nn.CrossEntropyLoss()
        else:
            print('unknown network in last layer')
            os._exit()


        # self.Q = Variable(Qpenalty*torch.eye(nx))
        # self.p = Variable(torch.zeros(nx))

        # if args.toGPU:
        #     self.Q = self.Q.cuda()
        #     self.p = self.p.cuda()
        """
        # G will be a matrix defined so:
        G = [ 1  0 0 0 0 0]
            [ 0  1 0 0 0 0]
            [ 0  0 1 0 0 0]
            [ 0  0 0 1 0 0]
            [ 0  0 0 0 1 0]
            [ 0  0 0 0 0 1]
            [-1  0 0 0 0 0]
            [ 0 -1 0 0 0 0]
            [ 0 0 -1 0 0 0]
            [ 0 0 0 -1 0 0]
            [ 0 0 0 0 -1 0]
            [ 0 0 0 0 0 -1]

        """
        # G = torch.eye(nineq, nx);         G = G.cuda() if args.toGPU else G

        # for i in range(nz):
        #     G[i][i] *= -1
        # self.G = Variable(G)

        # self.h = torch.ones(nineq);       self.h = self.h.cuda() if args.toGPU else self.h
        # self.h = Variable(self.h)

        # e = torch.Tensor();               e = e.cuda() if args.toGPU else e
        # self.e = Variable(e)

        def qp_layer(x):
            '''
            Parameters:
              Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
              p:  A (nBatch, nz) or (nz) Tensor.
              G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
              h:  A (nBatch, nineq) or (nineq) Tensor.
              A:  A (nBatch, neq, nz) or (neq, nz) Tensor. None in this case.
              b:  A (nBatch, neq) or (neq) Tensor. None in this case.

            Returns: \hat z: a (nBatch, nz) Tensor.
            '''
            nBatch = x.size(0)
            Q = self.Q
            p = x.view(nBatch, -1)
            G = self.G
            h = self.h
            e = self.e
            x = QPFunction()(Q, p, G, h, e, e); x = x.cuda() if self.args.toGPU else x
            return x
        self.qp_layer = qp_layer

    def forward(self, x):
        
        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, self.batchSize, self.nHidden[0]).cuda()) 
        c0 = Variable(torch.zeros(self.num_layers, self.batchSize, self.nHidden[0]).cuda())

        h1 = Variable(torch.zeros(self.num_layers, self.batchSize, self.nHidden[1]).cuda()) 
        c1 = Variable(torch.zeros(self.num_layers, self.batchSize, self.nHidden[1]).cuda())

        h2 = Variable(torch.zeros(self.num_layers, self.batchSize, self.nHidden[2]).cuda()) 
        c2 = Variable(torch.zeros(self.num_layers, self.batchSize, self.nHidden[2]).cuda())

        # Forward propagate RNN layer 1
        # we could do this out, _ = self.lstm1(x, (h0, c0)) but James Bradbury tells me it'd be incorrect
        # out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm1(x)

        # Forward propagate RNN layer 2
        out, _ = self.lstm2(out)

        # Forward propagate RNN layer 2
        out, _ = self.lstm3(out)

        # Decode hidden batch of last time step     
        # out = self.fc(out[:, -1, :]) 
        out = self.fc(out[-1, :, :])   

        #Now add QP Layer
        # out = out.view(nBatch, -1)

        # out = self.qp_layer(out)

        return out
