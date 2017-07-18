#!/usr/bin/env python3
# coding=utf-8

'''
    Olalekan Ogunmolu 
    July 2017
'''
from __future__ import print_function
import os
import sys
import model
import argparse

# try: import setGPU
# except ImportError: pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import numpy.random as npr

import sys
sys.path.insert(0, "utils")
sys.path.insert(1, "ros")

#custom utility functions
try:
    # from utils.data_parser import loadSavedMatFile
    from data_parser import split_data
    from ros_comm import Listener
except ImportError: pass

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import rospy
import roslib
roslib.load_manifest('pyrnn')

from ensenso.msg import ValveControl
from geometry_msgs.msg import Pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--gpu', type=int,  default=0)
    parser.add_argument('--noutputs', type=int, default=3)
    parser.add_argument('--display', type=int,  default=1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--toGPU', type=bool,    default=True)
    parser.add_argument('--maxIter', type=int,  default=1000)
    parser.add_argument('--silent', type=bool,  default=True)
    parser.add_argument('--sim', type=bool,  default=False)
    parser.add_argument('--useVicon', type=bool, default=True)
    parser.add_argument('--save', type=bool, default='true')
    parser.add_argument('--model', type=str,default= 'lstm')
    parser.add_argument('--qpenalty', type=float, default=0.1)
    parser.add_argument('--real_net', type=bool,default=True, help='use real-time network approximator')
    parser.add_argument('--seed', type=int,default=123)
    parser.add_argument('--rnnLR', type=float,default=5e-3)
    parser.add_argument('--Qpenalty', type=float, default=0.1)
    parser.add_argument('--hiddenSize', type=list, nargs='+', default='966')
    args = parser.parse_args()

    models_dir = 'models'
    if not models_dir in os.listdir(os.getcwd()):
        os.mkdir(models_dir)   # path to store models
    args.models_dir = os.getcwd() + '/' + models_dir

    nFeatures, nCls, nHidden = 6, 3, list(map(int, args.hiddenSize))

    #Global Hyperparams
    numLayers = 1
    inputSize, sequence_length = 9, 9
    noutputs, batchSize = 6, args.batchSize

    # QP Hyperparameters
    '''
    6 valves = nz = 6
    no equality contraints neq = 0
    due to double sided inequliaties and introduction of
    slack variables, nineq = 12
    QPenalty is arbitrarily chosen. Ideally, set to 1
    '''
    nz, neq, nineq, QPenalty = 6, 0, 12, args.qpenalty
    net = model.LSTMModel(nz, neq, nineq, QPenalty,
                          inputSize, nHidden, batchSize, noutputs, numLayers)

    if args.toGPU:
        net = net.cuda()

    npr.seed(1)

    if args.sim:
        trainX, trainY, testX, testY = split_data("data/data.mat")
        train_in, train_out, test_in, train_out = split_data("data/data.mat")

    optimizer = optim.SGD(net.parameters(), lr=args.rnnLR)
    train(args, net, optimizer)

def train(args, net, optimizer):
    batchSize = args.batchSize
    iter,lr = 0, args.rnnLR
    num_epochs =  args.maxIter    

    l = Listener(Pose, ValveControl)
    for epoch in range(num_epochs):            

        inputs, labels = exportsToTensor(l.pose_export, l.controls_export)

        inputs = inputs.cuda() if args.toGPU else None            
        labels = labels.cuda() if args.toGPU else None

        # Forward 
        optimizer.zero_grad()
        outputs = net(inputs)

        # Backward 
        loss    = net.criterion(outputs, labels)
        loss.backward()

        # Optimize
        optimizer.step()

        if (epoch % 5) == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, epoch+batchSize, inputs.size(0),
                float(iter+batchSize)/inputs.size(0)*100,
            loss.data[0]))

    torch.save(net.state_dict(), args.models_dir + '/' + 'lstm_net_' + str(args.maxIter) + '.pkl') if args.save else None
        

def exportsToTensor(pose, controls):
    seqLength, outputSize = 5, 6
    inputs = torch.Tensor([[
                            controls.get('lo', 0), controls.get('bo', 0),
                            controls.get('bi', 0), controls.get('li', 0),
                            controls.get('ro', 0), controls.get('ri', 0),
                            pose.get('z', 0), pose.get('pitch', 0),
                            pose.get('yaw', 0)
                        ]])
    inputs = Variable((torch.unsqueeze(inputs, 1)).expand_as(torch.LongTensor(seqLength,1,9)))

    #will be [torch.FloatTensor of size 1x3]
    targets = (torch.Tensor([[
                            pose.get('z', 0), pose.get('z', 0),
                            pose.get('pitch', 0), pose.get('pitch', 0),
                            pose.get('roll', 0), pose.get('roll', 0)
                            ]])).expand(seqLength, 1, outputSize)
    targets = Variable(targets)

    return inputs, targets

if __name__ == '__main__':
        main()
