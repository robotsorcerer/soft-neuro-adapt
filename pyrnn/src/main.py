#!/usr/bin/env python2
# coding=utf-8

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
sys.path.insert(1, "ros")

#custom utility functions 
try:    
    # from utils.data_parser import loadSavedMatFile
    from data_parser import split_data
    from ros_comm import Listener
except Exception, e:
    raise e
    print('No Import of Utils')
    print 'No import of ROSCommEmulator'

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose

def print_header(msg):
    print('===>', msg)

def main(epoch, trainX, trainY): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--gpu', type=int,  default=0)
    parser.add_argument('--noutputs', type=int, default=3)
    parser.add_argument('--display', type=int,  default=1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--cuda', type=bool,    default=False)
    parser.add_argument('--maxIter', type=int,  default=10000)
    parser.add_argument('--silent', type=bool,  default=True)
    parser.add_argument('--sim', type=bool,  default=False)
    parser.add_argument('--useVicon', type=bool, default=True)
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--squash', type=bool,default= True)
    parser.add_argument('--model', type=str,default= 'lstm')
    parser.add_argument('--real_net', type=bool,default=True, help='use real-time network approximator')
    parser.add_argument('--seed', type=int,default=123)
    parser.add_argument('--rnnLR', type=float,default=5e-3)
    parser.add_argument('--hiddenSize', type=list, nargs='+', default='966')
    optnetP = subparsers.add_parser('optnet')
    optnetP.add_argument('--Qpenalty', type=float, default=0.1)
    args = parser.parse_args()
    #args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '{}'.format(args.model)
    if args.model == 'optnet':
        t += '.Qpenalty={}'.format(args.Qpenalty)
    setproctitle.setproctitle('lekan.soft-robot.' + t)

    nFeatures, nCls, nHidden = 6, 3, map(int, args.hiddenSize)
    #ineq constraints are [12 x 3] in total
    # print('model ', model)

    #hyperparams
    inputSize = 9
    numLayers = 2
    sequence_length = 9
    noutputs = 3
    batchSize = args.batchSize

    net = model.LSTMModel(inputSize, nHidden, batchSize, noutputs, numLayers)

    if args.cuda:
        net = net.cuda()

    # print_header
    ('Building model')

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

    if args.sim:
        trainX, trainY, testX, testY = split_data("data/data.mat")
        train_in, train_out, test_in, train_out = split_data("data/data.mat")

    optimizer = optim.SGD(net.parameters(), lr=args.rnnLR)

    train(args, net, epoch, optimizer, trainX, trainY)

def train(args, net, epoch, optimizer, trainX, trainY):
    batchSize = args.batchSize  
    iter,lr = 0, args.rnnLR 
    num_epochs = 500

    batch_data_t = torch.FloatTensor(batchSize, trainX.size(1))
    batch_targets_t = torch.FloatTensor(batchSize, trainY.size(1))

    if args.cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()

    batch_data = Variable(batch_data_t, requires_grad=False)
    batch_targets = Variable(batch_targets_t, requires_grad=False)
    
    # for epoch in range(num_epochs):
    inputs = trainX     #Variable(torch.Tensor(5, 1, 9))
    labels = trainY     #Variable(torch.Tensor(5, 3))

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = net(inputs)
    loss    = net.criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # if rospy.is_shutdown():
        # break

    if (epoch % 10) == 0:
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
            epoch, epoch+batchSize, trainX.size(0),
            float(iter+batchSize)/trainX.size(0)*100,
            loss.data[0]))
    
def writeParams(args, model, tag):
    if args.model == 'optnet':
        A = model.A.data.cpu().numpy()
        np.savetxt(os.path.join(args.save, 'A.{}'.format(tag)), A)

def exportsToTensor(pose, controls):
    #will be [torch.FloatTensor of size 1x9]
    inputs = torch.Tensor([[
                            controls.get('lo', 0), controls.get('bo', 0),
                            controls.get('bi', 0), controls.get('li', 0), 
                            controls.get('ro', 0), controls.get('ri', 0),
                            pose.get('z', 0), pose.get('pitch', 0), 
                            pose.get('yaw', 0)
                        ]])
    inputs = Variable(
                     (torch.unsqueeze(inputs, 1)).expand_as(torch.LongTensor(5,1,9))
                     )

    #will be [torch.FloatTensor of size 1x3]
    targets = (torch.Tensor([[                            
                            pose.get('z', 0), pose.get('pitch', 0), 
                            pose.get('yaw', 0)
                            ]])).expand(5, 3)
    targets = Variable(targets)
    return inputs, targets

    
if __name__ == '__main__':

    l = Listener(Pose, Twist)
    r = rospy.Rate(10)
    epoch = 0
    while not rospy.is_shutdown():

        trainX, trainY = exportsToTensor(l.pose_export, l.controls_export)
        # print l.controls_export.get('lo', None)
        # print l.pose_export.get("pitch", None)
        # print trainX
        epoch += 1
        main(epoch, trainX, trainY)
        r.sleep()
        pass