import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, nFeatures, nCls, bn, nineq=200, neq=0, eps=1e-4, 
                noutputs=3,numLayers=1, nHidden=[9, 6, 6]):
        super().__init__()

        self.cost = nn.MSELoss()
        self.nHidden = nHidden
        self.noutputs = noutputs
        # self.neunet = nn.Sequential()
        self.lstm1  = nn.LSTM(nHidden[0],nHidden[0],num_layers=numLayers)
        self.drop1  = nn.Dropout(0.3)
        self.lstm2  = nn.LSTM(nHidden[0],nHidden[1],num_layers=numLayers)
        self.drop2  = nn.Dropout(0.3)
        self.lstm3  = nn.LSTM(nHidden[1],nHidden[2],num_layers=numLayers),
        self.drop3  = nn.Dropout(0.3),
        self.output = nn.Linear(nHidden[2], noutputs)

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-QP-FC-Softmax
        # LSTM-dropout-LSTM-dropout-lstm-dropout-QP-FC
        x = x.view(nBatch, -1)
        x = self.lstm1(x)
        x = self.drop1(x)
        x = self.lstm2(x)
        x = self.drop2(x)
        x = self.lstm3(x)
        x = self.drop3(x)

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -x.view(nBatch,-1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))

        x = QPFunction(verbose=False)(p.double(), Q, G, h, A, b).float()
        x = self.fc2(x)

        return F.log_softmax(x)

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

# rnn = nn.LSTM(10,20,2)
# input = Variable(torch.randn(5,3, 10))
# h0 = Variable(torch.randn(2,3, 20))
# c0 = Variable(torch.randn(2,3,20))
# output, hn = rnn(input, (h0, c0))

# print ('output', output)
# print('hn ', hn)

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

# test_LSTM_cell()

# lstm_model()

model = LSTMModel()
model.lstm_model()

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