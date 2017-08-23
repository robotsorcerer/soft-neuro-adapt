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
import copy
import logging
import argparse

import numpy as np
import scipy as sp
import numpy.random as npr

import rospy, rospkg

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from os.path import dirname, abspath#, join, sep

icra_path = dirname(dirname(abspath(__file__)))
base_path = icra_path + '/../'

sys.path.append(icra_path)
sys.path.append(base_path)

from pyrnn import split_csv_data

from utils import GMM, finite_differences, approx_diff
from gmr_lib import emInitKmeans, EM, gaussPDF, GMR, plotGMM
from dynamics import DynamicsPriorGMM

LOGGER = logging.getLogger(__name__)


# Ported from Chelsea Finn's code

def compute_clusters(traj_data, n_clusters):

    # data into kmeans:     D x N array representing N datapoints of D dimensions.
    # Note that fdata is transposed within the KMeans file
    priors, mu, sigma = emInitKmeans(traj_data.T, n_clusters)

    return priors, mu, sigma


def main():
    parse = argparse.ArgumentParser(description="options from terminal")
    parse.add_argument("--verbose", action='store_true')
    parse.add_argument("--K", type=int, default=6, help="number of kmeans clusters")
    args = parse.parse_args()

    rospack = rospkg.RosPack()
    pyrnn_path = rospack.get_path('pyrnn')
    data_file  = pyrnn_path + "/" + "src/data/training_data.csv.gz"

    train, test  = split_csv_data(data_file)
    # convert torch Variable to numpy arrays for gmm clusters
    for k, v in train.items():
        train[k] = train[k].data.numpy()
    for k, v in test.items():
        test[k] = test[k].data.numpy()
    # Now concatenate in and out for clfdm. Remember to remove outs
    input_data = np.r_[train['in'][:,:6], test['in'][:,:6]]
    output_data = np.r_[train['out'], test['out']]

    # compute the derivatives of the time-series data
    test_input  = np.copy(input_data)
    test_out    = np.copy(output_data)

    input_grad  = approx_diff(input_data, delay=2, dt=1e-5)
    output_grad = approx_diff(output_data, delay=2, dt=1e-5)
    # print('nan: ', np.where(input_grad==np.nan), '\n\ninf: ', np.where(input_grad==np.inf))

    traj_data = np.c_[
                        input_data, output_data
                        # npr.shuffle(np.r_[input_data, output_data]),
                        # npr.shuffle(np.r_[input_grad, output_grad])
                    ]


    try:
        # n_clusters = args.K
        # priors0, mu0, sigma0 = compute_clusters(traj_data, n_clusters)
        # must have updated prior with X and U first

        prior_gmm = DynamicsPriorGMM()
        print('input: {}, output: {}'.format(input_data.shape, output_data.shape))
        prior_gmm.update(input_data, output_data)

        mu0, Phi, m, n0 = prior_gmm.initial_state()
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An N x D array of points

        Returns cluster probabilities
        """
        # gmm = GMM()
        # gmm.K = n_clusters
        # gmm.sigma = sigma0
        # gmm.mu = mu0
        # logwts = gmm.clusterwts(traj_data)
        print(mu0, Phi, m, n0)

        # compute the expectation maximizatyion on the priors
        # priors0, mu0, sigma0 = EM(data, priors0, mu0, sigma0)

    except KeyboardInterrupt:
        print("shutting down ros")



if __name__ == "__main__":
    main()
