#!/usr/bin/env python2
# coding=utf-8
'''
    Olalekan Ogunmolu
    August 2017

    This code parameterizes the dataset gathered during the
    identification experiment with a mixtuire of N Gaussians

'''

from __future__ import print_function

import os, time
import sys
import logging

import numpy as np
import scipy as sp

import rospy, rospkg

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from os.path import dirname, abspath#, join, sep

icra_path = dirname(dirname(abspath(__file__)))
base_path = icra_path + '/../'

# sys.path.append(icra_path)
sys.path.append(base_path)

from pyrnn import split_csv_data

# from src.data
# print('base_path: ', base_path)


LOGGER = logging.getLogger(__name__)

def disp():
    try:
        # rospy.init_node('param_estimator_node', anonymous=True)
        # rate = rospy.Rate(5)
        rospack = rospkg.RosPack()
        pyrnn_path = rospack.get_path('pyrnn')
        data_file  = pyrnn_path + "/" + "src/data/training_data.csv.gz"

        train, test  = split_csv_data(data_file)
    except KeyboardInterrupt:
        print("shutting down ros")



if __name__ == "__main__":
    disp()
