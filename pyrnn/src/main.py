#!/usr/bin/env python2

import argparse
import csv
import os
import shutil
from tqdm import tqdm

try: import setGPU
except ImportError: pass

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import numpy.random as npr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import setproctitle

import model
import sys
sys.path.insert(0, "utils")
#custom utility functions 
try:    
    # from utils.data_parser import loadSavedMatFile
    from data_parser import split_data
except Exception, e:
    raise e
    print('No Import of Utils')

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def print_header(msg):
    print('===>', msg)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--batchSize', type=int, default=100)
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--gpu', type=int,  default=0)
    parser.add_argument('--noutputs', type=int, default=3)
    parser.add_argument('--display', type=int,  default=1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--cuda', type=bool,    default=True)
    parser.add_argument('--maxIter', type=int,  default=10000)
    parser.add_argument('--silent', type=bool,  default=True)
    parser.add_argument('--useVicon', type=bool, default=True)
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--squash', type=bool,default= True)
    parser.add_argument('--model', type=str,default= 'lstm')
    parser.add_argument('--real_time_net', type=bool,default=True, help='use real-time network approximator')
    parser.add_argument('--seed', type=int,default=123)
    parser.add_argument('--rnnLR', type=float,default=5e-3)
    parser.add_argument('--hiddenSize', type=list, nargs='+', default='966')
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    nFeatures, nCls, nHidden = 6, 3, map(int, args.hiddenSize)
    #ineq constraints are [12 x 3] in total
    # print('model ', model)

    print_header('Building model')
    net = model.LSTMModel(nFeatures, nCls, nHidden)

    if args.cuda:
        net = net.cuda()

    t = '{}'.format('optnet')
    if args.save is None:
        args.save = os.path.join(args.work, t)

    save = args.save
    if os.path.isdir(save):
        shutil.rmtree(save)
        os.makedirs(save)
    
    npr.seed(1)

    fields = ['epoch', 'loss', 'err']
    trainF = open(os.path.join(save, 'train.csv'), 'w')
    trainW = csv.writer(trainF)
    trainW.writerow(fields)
    trainF.flush()
    fields = ['epoch', 'loss', 'err']
    testF = open(os.path.join(save, 'test.csv'), 'w')
    testW = csv.writer(testF)
    testW.writerow(fields)
    testF.flush()

    trainX, trainY, testX, testY = split_data("data/data.mat")

    writeParams(args, net, 'init')
    # test(args, 0, net, testF, testW, testX, testY)
    # train_in, train_out, test_in, train_out = split_data("data/data.mat")
    optimizer = optim.Adam(net.parameters(), lr=args.rnnLR)
    train(args, net, optimizer, trainX, trainY, trainW, trainF)

def train(args, neunet, optimizer, trainX, trainY, trainW, trainF):
    batchSize = args.batchSize  
    iter,lr = 0, args.rnnLR 

    batch_data_t = torch.FloatTensor(batchSize, trainX.size(1))
    batch_targets_t = torch.FloatTensor(batchSize, trainY.size(1))

    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()

    batch_data = Variable(batch_data_t, requires_grad=False)
    batch_targets = Variable(batch_targets_t, requires_grad=False)

    for i in range(0, trainX.size(0), batchSize):
        batch_data[:] = trainX.data[i:i+batchSize]
        batch_targets[:] = trainY.data[i:i+batchSize]

        optimizer.zero_grad()
        preds = neunet(batch_data)
        loss = neunet.cost(preds, batch_targets)
        loss.backward()
        optimizer.step()

        err = get_nErr(preds.data, batch_targets.data)/batchSize
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} Err: {:.4f}'.format(
            epoch, i+batchSize, trainX.size(0),
            float(i+batchSize)/trainX.size(0)*100,
            loss.data[0], err))

        trainW.writerow((epoch-1+float(i+batchSize)/trainX.size(0), loss.data[0], err))
        trainF.flush()

def writeParams(args, model, tag):
    if args.model == 'optnet':
        A = model.A.data.cpu().numpy()
        np.savetxt(os.path.join(args.save, 'A.{}'.format(tag)), A)
    
if __name__ == '__main__':
    main()
