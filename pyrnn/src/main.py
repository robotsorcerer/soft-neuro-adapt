#!/usr/bin/env python2
# coding=utf-8
'''
    Olalekan Ogunmolu
    July 2017

    This code subscribes to vicon/ensenso messages, parameterizes
    the lagged input vector to the neural network and then sends
    control torques to the valves.
'''

from __future__ import print_function
import os
import csv
import sys
import time
import math
import shutil
import socket
import argparse
import threading
from itertools import count
from datetime import datetime

import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

try: import setGPU
except ImportError: pass


sys.path.insert(0, "utils")
sys.path.insert(1, "ros")

#custom utility functions
try:
    from data_parser import split_mat_data, split_csv_data
    from ros_comm import Listener
except ImportError: pass

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import rospy
import rospkg
import threading
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import model
import early_stopping as es
# sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from pyrnn.src.utils import model
# from pyrnn.src.utils import early_stopping as es
from ensenso.msg import ValveControl
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray #as Float64MultiArray
from std_msgs.msg import MultiArrayDimension as MultiDim

torch.set_default_tensor_type('torch.DoubleTensor')
npr.seed(1)

class Net(Listener):
    """
        Trains the adaptive model following control neural network
    """
    def __init__(self, args):
        Listener.__init__(self, Pose, ValveControl)
        self.args = args

        #Global Hyperparams
        noutputs, batchSize = 3, args.batchSize
        numLayers, inputSize, sequence_length = 1, 9, 36
        nFeatures, nCls = 6, 3
        nHidden = [18, 6, 3] #list(map(int, args.hiddenSize))

        # QP Hyperparameters
        self.net = model.LSTMModel(args, inputSize, nHidden, batchSize, noutputs, numLayers)

        if args.test:
            self.net.load_state_dict(torch.load('models/' + args.model))
            self.net.eval()

        if not args.toGPU:
            self.net.cpu()

        # handler for class publisher
        self.weights_pub = rospy.Publisher('/mannequine_pred/net_weights', Float64MultiArray, queue_size=10)
        self.biases_pub = rospy.Publisher('/mannequine_pred/net_biases', Float64MultiArray, queue_size=10)
        self.net_control_law_pub = rospy.Publisher('/mannequine_pred/preds', ValveControl, queue_size=10)

        # GPU object mover
        self.net = self.net.cuda() if args.toGPU else self.net
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.rnnLR)
        self.scheduler = es.EarlyStop(self.optimizer, 'min')
        self.filename = 'train_data.txt'    # filename of training data
        npr.seed(1)

        # save params
        save = self.args.save
        if self.args.save is None:
            t = os.path.join(os.getcwd(), save)
        if os.path.isdir(save):
            shutil.rmtree(save)
            # shutil.move(save, save + ".old" + '_' +  datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')  )
        os.makedirs(save)

        self.save = save


    def exportsToTensor(self, pose, controls):
        seqLength, outputSize = 5, 6
        inputs = torch.Tensor([[
                                 controls.get('li', 0), controls.get('lo', 0),
                                 controls.get('bi', 0), controls.get('bo', 0),
                                 controls.get('ri', 0), controls.get('ro', 0),
                                 pose.get('roll', 0), pose.get('z', 0), pose.get('pitch', 0)
                             ]])
        inputs = Variable((torch.unsqueeze(inputs, 1)).expand_as(torch.LongTensor(seqLength,1,9)))

        #will be [torch.FloatTensor of size 1x3]
        targets = (torch.Tensor([[
                                 pose.get('roll', 0), pose.get('roll', 0),
                                 pose.get('z', 0), pose.get('z', 0),
                                 pose.get('pitch', 0), pose.get('pitch', 0),
                             ]])).expand(seqLength, 1, outputSize)
        targets = Variable(targets)

        return inputs, targets

    def validate(self):
        inputs, targets = self.exportsToTensor(self.get_pose(), self.get_controls())

        inputs = inputs.cuda() if self.args.toGPU else None
        targets = targets.cuda() if self.args.toGPU else None

        # Forward
        self.optimizer.zero_grad()
        outputs = self.net(inputs)

        # Backward
        val_loss    = self.net.criterion(outputs, targets)
        val_loss.backward()

        # Optimize
        self.optimizer.step()

        return val_loss

    def validate_offline(self, idx):
        inputs, targets = self.val_inputs, self.val_targets

        inputs = inputs.cuda() if self.args.toGPU else None
        targets = targets.cuda() if self.args.toGPU else None

        # Forward
        self.optimizer.zero_grad()
        outputs = self.net(inputs)

        # Backward
        val_loss    = self.net.criterion(outputs, targets)
        val_loss.backward()

        # Optimize
        self.optimizer.step()

        return val_loss

    def tensorToMultiArray(self, tensor, string):
        msg = Float64MultiArray()  
        msg.data = tensor.resize_(torch.numel(tensor))
        for i in range(tensor.dim()):
            dim_desc = MultiDim
            dim_desc.size = tensor.size(i)
            dim_desc.stride = tensor.stride(i)
            dim_desc.label = string
            # msg.layout.dim.append(dim_desc)
        return msg

    def offline_train(self, train):          
        #Global Hyperparams  
        seqLength   = 5
        numLayers   = 1
        outputSize  = train['out'].size(-1)
        inputSize   = train['in'].size(-1)
        print('inputs size: ', inputSize)
        print('outputs size: ', outputSize)
        batchSize   = 50
        nHidden     = [int(outputSize/2), 6, outputSize] 

        net = model.LSTMModel(self.args, inputSize, nHidden, batchSize, outputSize, numLayers)
        net = net.cuda() if self.args.toGPU else net
        optimizer = optim.Adam(net.parameters(), lr=self.args.rnnLR)
        criterion  = nn.MSELoss(size_average=False)

        train_dataset = data.TensorDataset(train['in'].data, train['out'].data)
        train_loader  = data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False)

        for idx in count(1): 

            loss = None # reset loss
            for epoch, (inputs, targets) in enumerate(train_loader):

                inputs = Variable(inputs)
                targets = Variable(targets)

                if inputs.size(0) != batchSize:
                    break
                inputs_  = inputs.expand_as( torch.LongTensor(seqLength, batchSize, inputSize ))

                inputs_ = inputs_.cuda() if self.args.toGPU else None
                targets_ = targets.cuda() if self.args.toGPU else None

                # Forward
                optimizer.zero_grad()
                outputs = net(inputs_)
                # print('inputs size: ', type(inputs_))#.size())
                # print('outputs size: ', outputs.size())
                # print('targets size: ', targets_.size())

                # Backward
                loss    = criterion(outputs, targets_)
                loss.backward()

                # Optimize
                optimizer.step()

                # save train and val loss
                fields = ['epoch', 'train_loss']
                trainF = open(os.path.join(self.save, 'train_csv'), 'a')
                trainW = csv.writer(trainF)
                trainW.writerow([epoch, loss.data[0]])
                trainF.flush()


                if (epoch % 50) == 0:
                    print('Epoch: {}  | \ttrain loss: {:.4f}  '.format(
                        epoch, loss.data[0]))

                if self.args.adaptLR:
                    if  ((epoch % 100) == 0):
                        lr = 1./(epoch+self.args.rnnLR)
                    optimizer = optim.Adam(net.parameters(), lr=lr)

                if loss.data[0] < 2:
                    print("achieved nice convergence")

                    model_filename =  'lstm_net_' +  \
                            datetime.strftime(datetime.now(), '%m-%d-%y_%H::%M') + '.pkl'
                    print("saving model file as: ", model_filename)
                    torch.save(net.state_dict(), self.args.models_dir + '/' + model_filename) if self.args.save else None

                    break        
                if rospy.is_shutdown():
                    os._exit()
            if loss.data[0] < 2:
                break            


    def train(self):
        #plot params
        plt.ioff()
        plt.xlabel('time')
        plt.ylabel('mean square loss')
        fig, ax = plt.subplots()
        ax.legend(loc='lower right')

        for epoch in count(1): #range(num_epochs):

            self.listen()
            inputs, targets = self.exportsToTensor(self.get_pose(), self.get_controls())

            inputs = inputs.cuda() if self.args.toGPU else None
            targets = targets.cuda() if self.args.toGPU else None

            # Forward
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            # Backward
            loss    = self.net.criterion(outputs, targets)
            loss.backward()

            # Optimize
            self.optimizer.step()

            # validate the loss
            val_loss  = self.validate()
            # Implement early stopping. Note that step should be called after validate()
            self.scheduler.step(val_loss)

            net_biases =  self.net.fc.bias.data.cpu()
            net_weights = self.net.fc.weight.data.cpu()
            # publish net weights and biases
            biases_msg = self.tensorToMultiArray(net_biases, 'biases')
            weights_msg = self.tensorToMultiArray(net_weights, 'weights')

            # sample from the output of the trained network and update the control trajectories
            idx = npr.randint(0, outputs.size(0))  # randomly pick a row index in the QP layer weights
            control_action = outputs[idx,:].data  # convert Variable to data
            # $ take the mean of all probabilities along the time dimenbsion
            # control_action = outputs.mean(0).data.t()
            # control_action = outputs.multinomial()
            # print('control_action: ', control_action)
            control_msg = ValveControl()   # Control Message to valves
            control_msg.stamp = rospy.Time.now();       control_msg.seq = epoch
            control_msg.left_bladder_pos = control_action[0]; control_msg.left_bladder_neg = control_action[1];
            control_msg.base_bladder_pos = control_action[2]; control_msg.base_bladder_neg = control_action[3];
            control_msg.right_bladder_pos = control_action[4]; control_msg.right_bladder_neg = control_action[5];
            # print('control_law_msg: ', control_msg)

            # publish the weights
            self.biases_pub.publish(biases_msg)
            self.weights_pub.publish(weights_msg)
            self.net_control_law_pub.publish(control_msg)

            # save train and val loss
            fields = ['epoch', 'train_loss/val_loss']
            trainF = open(os.path.join(self.save, 'train_csv'), 'a')
            trainW = csv.writer(trainF)
            trainW.writerow([epoch, loss.data[0], val_loss.data[0]])
            trainF.flush()

            # TODO: GUI Not correctly implemented
            if self.args.display:# show some plots
                plt.ion()
                line1, = ax.plot(val_loss.data[0], 'b--', linewidth=2.5, label='validation loss')
                line2, = ax.plot(loss.data[0], 'r--', linewidth=2.5, label="training loss")
                plt.draw()
                plt.grid(True)

            if (epoch % 5) == 0:
                print('Epoch: {}  | \ttrain loss: {:.4f}  | \tval loss: {:.4f}'.format(
                    epoch, loss.data[0], val_loss.data[0]))
            if self.args.adaptLR and ((epoch % 100) == 0):
                lr = 1./epoch
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            if val_loss.data[0] < 5e-3:
                print("achieved nice convergence")
                break

            if rospy.is_shutdown():
                os._exit()
        model_filename =  'lstm_net_' +  \
                datetime.strftime(datetime.now(), '%m-%d-%y_%H::%M') + '.pkl'
        print("saving model file as: ", model_filename)
        torch.save(self.net.state_dict(), self.args.models_dir + '/' + model_filename) if self.args.save else None

    def test(self):
        #plot params
        plt.ioff()
        plt.xlabel('time')
        plt.ylabel('mean square loss')
        fig, ax = plt.subplots()
        ax.legend(loc='lower right')

        for epoch in count(1): #range(num_epochs):

            self.listen()
            inputs, targets = self.exportsToTensor(self.get_pose(), self.get_controls())

            inputs = inputs.cuda() if self.args.toGPU else None
            targets = targets.cuda() if self.args.toGPU else None

            # Forward
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            # Backward
            loss    = self.net.criterion(outputs, targets)
            loss.backward()

            # Optimize
            self.optimizer.step()

            net_biases =  self.net.fc.bias.data.cpu()
            net_weights = self.net.fc.weight.data.cpu()
            # publish net weights and biases
            biases_msg = self.tensorToMultiArray(net_biases, 'biases')
            weights_msg = self.tensorToMultiArray(net_weights, 'weights')

            # sample from the output of the trained network and update the control trajectories
            idx = npr.randint(0, outputs.size(0))  # randomly pick a row index in the QP layer weights
            control_action = outputs[idx,:].data  # convert Variable to data
            control_msg = ValveControl()   # Control Message to valves
            control_msg.stamp = rospy.Time.now();       control_msg.seq = epoch
            control_msg.left_bladder_pos = control_action[0]; control_msg.left_bladder_neg = control_action[1]
            control_msg.right_bladder_pos = control_action[2]; control_msg.right_bladder_neg = control_action[3]
            control_msg.base_bladder_pos = control_action[4]; control_msg.base_bladder_neg = control_action[5]

            # publish the weights
            self.biases_pub.publish(biases_msg)
            self.weights_pub.publish(weights_msg)
            self.net_control_law_pub.publish(control_msg)

            # save test loss
            fields = ['epoch', 'test_loss']
            testF = open(os.path.join(self.save, 'test_csv'), 'a')
            testW = csv.writer(testF)
            testW.writerow([epoch, loss.data[0]])
            testF.flush()

            # TODO: GUI Not correctly implemented
            if self.args.display:# show some plots
                plt.ion()
                line1, = ax.plot(val_loss.data[0], 'b--', linewidth=2.5, label='validation loss')
                line2, = ax.plot(loss.data[0], 'r--', linewidth=2.5, label="training loss")
                plt.draw()
                plt.grid(True)

            print('Epoch: {}  | \ttest loss: {:.4f} '.format(
                epoch, loss.data[0]))
            if self.args.adaptLR and ((epoch % 100) == 0):
                lr = 1./epoch
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

            if rospy.is_shutdown():
                os._exit()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--gpu', type=int,  default=0)
    parser.add_argument('--noutputs', type=int, default=3)
    parser.add_argument('--display', type=int,  default=0)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--toGPU', type=bool,    default=True)
    parser.add_argument('--maxIter', type=int,  default=1000)
    parser.add_argument('--silent', type=bool,  default=True)
    parser.add_argument('--adaptLR', type=bool,  default=True)
    parser.add_argument('--sim', type=bool,  default=False)
    parser.add_argument('--useVicon', type=bool, default=True)
    parser.add_argument('--lastLayer', type=str, default='linear')
    parser.add_argument('--save', type=str, default='results')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--identify', action='store_true', default=False)
    parser.add_argument('--model', type=str,default= 'lstm_net_07-21-17_16::09.pkl')
    parser.add_argument('--qpenalty', type=float, default=1.0)
    parser.add_argument('--real_net', type=bool,default=True, help='use real-time network approximator')
    parser.add_argument('--seed', type=int,default=123)
    # parser.add_argument('--save', type=str, default='results')
    parser.add_argument('--rnnLR', type=float,default=5e-3)
    parser.add_argument('--Qpenalty', type=float, default=0.1)
    parser.add_argument('--hiddenSize', type=list, nargs='+', default='963')
    args = parser.parse_args()
    print(args) if args.verbose else None
    # time.sleep(50)

    models_dir = 'models'
    if not models_dir in os.listdir(os.getcwd()):
        os.mkdir(models_dir)   # path to store models
    args.models_dir = os.getcwd() + '/' + models_dir

    if args.sim:
        trainX, trainY, testX, testY = split_data("data/data.mat")
        train_in, train_out, test_in, train_out = split_data("data/data.mat")

    if args.offline:
        train, test  = split_csv_data("data/training_data.csv.gz")

        # test_dataset = data.TensorDataset(test['in'], test['out'])
        # test_loader  = data.DataLoader(test_dataset,
        #                     batch_size=batchSize, shuffle=False)


    rospy.init_node('pose_control_listener', anonymous=True)
    rate = rospy.Rate(5)
    try:
        if not rospy.is_shutdown():
            network = Net(args)

            if args.identify:
                network.identify_model()
            if args.offline:
                network.offline_train(train)
                # network.offline_test(test)
            if args.test:
                network.test()
            rate.sleep()
    except KeyboardInterrupt:
        print("shutting down ros")

if __name__ == '__main__':
    main()
