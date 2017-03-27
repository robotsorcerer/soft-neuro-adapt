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


#hyperparams
inputSize = 9
nHidden   = 9
numLayers = 2
sequence_length = 9
num_epochs = 500
noutputs = 3
batchSize = 1

# Loss and Optimizer
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.1)

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
    '''
    def __init__(self, inputSize, nHidden, batchSize, noutputs=3, numLayers=2):
        super(LSTMModel, self).__init__()
        
        self.cost = nn.MSELoss(size_average=False)
        self.noutputs = noutputs
        self.num_layers = numLayers
        self.inputSize = inputSize
        self.nHidden = nHidden
        self.batchSize = batchSize
        self.noutputs = noutputs
        
        #define recurrent and linear layers
        self.lstm1  = nn.LSTM(inputSize,nHidden[0],num_layers=numLayers)
        self.lstm2  = nn.LSTM(nHidden[0],nHidden[1],num_layers=numLayers)
        self.lstm3  = nn.LSTM(nHidden[1],nHidden[2],num_layers=numLayers)
        self.fc     = nn.Linear(nHidden[2], noutputs)
        self.drop   = nn.Dropout(0.3)

    def forward(self, x):
        # Set initial states 
        h0 = Variable(torch.Tensor(self.num_layers, batchSize, self.nHidden[0])) 
        c0 = Variable(torch.Tensor(self.num_layers, batchSize, self.nHidden[0]))        
        # Forward propagate RNN layer 1
        out, _ = self.lstm1(x, (h0, c0)) 
        out = self.drop(out)
        
        # Set hidden layer 2 states 
        h1 = Variable(torch.Tensor(self.num_layers, batchSize, self.nHidden[1])) 
        c1 = Variable(torch.Tensor(self.num_layers, batchSize, self.nHidden[1]))        
        # Forward propagate RNN layer 2
        out, _ = self.lstm2(out, (h1, c1))  
        
        # Set hidden layer 3 states 
        h2 = Variable(torch.Tensor(self.num_layers, batchSize, self.nHidden[2])) 
        c2 = Variable(torch.Tensor(self.num_layers, batchSize, self.nHidden[2]))        
        # Forward propagate RNN layer 2
        out, _ = self.lstm3(out, (h2, c2)) 
        out = self.drop(out) 
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  
        return out

lstm = LSTMModel(inputSize, nHidden, batchSize, noutputs, numLayers)
#lstm.cuda()

# Loss and Optimizer
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.1)

#images and labels
for epoch in range(num_epochs):
    inputs = Variable(torch.Tensor(5, 1, 9))
    labels = Variable(torch.Tensor(5, 3))

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = lstm(inputs)
    loss    = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch % 10) == 0:
        print ('Epoch [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, loss.data[0]))








class unusedFunctions():

    def __init__():

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

#hyperparams
inputSize = 1
nHidden   = 9
numLayers = 2
sequence_length = 9
num_epochs = 500
noutputs = 3

lstm = LSTMModel(inputSize, nHidden, noutputs=3, numLayers=1)
lstm.cuda()

# Loss and Optimizer
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.1)

#images and labels
images = torch.Tensor(1, 9, 3)
labels = torch.Tensor(3)