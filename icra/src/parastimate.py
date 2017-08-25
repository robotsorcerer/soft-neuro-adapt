#!/usr/bin/env python3
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

import rospy
import rospkg

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from os.path import dirname, abspath#, join, sep

icra_path = dirname(dirname(abspath(__file__)))
base_path = icra_path + '/../'

sys.path.append(icra_path)
sys.path.append(base_path)

from pyrnn import split_csv_data

from utils import GMM, IterationData, TrajectoryInfo
from gmr_lib import emInitKmeans, EM, gaussPDF, GMR, plotGMM
from dynamics import DynamicsPriorGMM, DynamicsLRPrior
from config import DYN_PRIOR_GMM, DYNAMICS_PROPERTIES

LOGGER = logging.getLogger(__name__)


# Ported from Chelsea Finn's code

def compute_clusters(traj_data, n_clusters):

    # data into kmeans:     D x N array representing N datapoints of D dimensions.
    # Note that fdata is transposed within the KMeans file
    priors, mu, sigma = emInitKmeans(traj_data.T, n_clusters)

    return priors, mu, sigma

def populate_data():
    rospack = rospkg.RosPack()
    pyrnn_path = rospack.get_path('pyrnn')
    data_file  = pyrnn_path + "/" + "src/data/training_data.csv.gz"

    train, test  = split_csv_data(data_file)
    np.set_printoptions(precision=4)
    # convert torch Variable to numpy arrays for gmm clusters
    for k, v in train.items():
        train[k] = train[k].data.numpy()
    for k, v in test.items():
        test[k] = test[k].data.numpy()
    # Now concatenate in and out for clfdm. Remember to remove outs
    input_data =  np.nan_to_num(np.r_[train['in'][:,:6], test['in'][:,:6]], copy=False)
    output_data = np.nan_to_num(np.r_[train['out'], test['out']], copy=False)

    # compute the derivatives of the time-series data
    test_input  = np.copy(input_data)
    test_out    = np.copy(output_data)

    # input_grad  = approx_diff(input_data, delay=2, dt=1e-5)
    # output_grad = approx_diff(output_data, delay=2, dt=1e-5)
    # print('nan: ', np.where(input_grad==np.nan), '\n\ninf: ', np.where(input_grad==np.inf))

    traj_data = np.c_[
                        test_input, test_out
                        # npr.shuffle(np.r_[test_input, test_output]),
                        # npr.shuffle(np.r_[input_grad, output_grad])
                    ]
    # print('input: {}, output: {}'.format(test_input.shape, test_out.shape))
    T = DYN_PRIOR_GMM['T'] # index of time for states and controls
    U_full = np.tile(np.expand_dims(test_input, axis=1),  [1, T, 1])#[:30,:,:]
    # print(U.shape)
    X_full = np.tile(np.expand_dims(test_out, axis=1), [1, T, 1])#[:30,:,:]

    return X_full, U_full


def main():
    parse = argparse.ArgumentParser(description="options from terminal")
    parse.add_argument("-v", "--verbose", action='store_true')
    parse.add_argument("-s", "--silent", action='store_true')
    parse.add_argument("-K",  "--clusters", type=int, default=6, help="number of kmeans clusters")
    args = parse.parse_args()

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    cond_ = DYN_PRIOR_GMM['initial_condition']
    n_clusters = DYN_PRIOR_GMM['max_clusters']
    sample_size = DYN_PRIOR_GMM['sample_size']

    X_full, U_full = populate_data()

    print('X full shape: {}, U full shape: {}'.format(X_full.shape, U_full.shape))
    def sample(chunk_size):
        npr.shuffle(X_full)
        npr.shuffle(U_full)
        return X_full[:chunk_size,:,:], U_full[:chunk_size,:,:]

    try:
        # IterationData objects for each condition.
        cur = [IterationData() for _ in range(cond_)]
        prev = [IterationData() for _ in range(cond_)]

        for m in range(cond_):
            cur[m].traj_info = TrajectoryInfo() #L24 algorithm_utils.py
            cur[m].traj_info.dynamics = DynamicsLRPrior(DYN_PRIOR_GMM)

        # prior = DynamicsLRPrior(DYNAMICS_PROPERTIES)
        for m in range(cond_):
            X, U = sample(sample_size)
            """
             update prior and fit dynamics
             update prior gets samples from the robot,
             and computes the prior. We degine a temporal evolution
             of the state dynamics by forming the vector of samples [x_t, u_t, x_{t+1}]

             o we pick the number of clusters by starting with 2
             and we increase K until the maximum likelihood estimate of the GMM converges
              on a
             #L20 dynamics_lr_prior
            """
            cur[m].traj_info.dynamics.update_prior(X, U)
            cur[m].traj_info.dynamics.fit(X, U)  #L31 dynamics_lr_prior

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            cur[m].traj_info.x0mu = x0mu
            cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           DYNAMICS_PROPERTIES['initial_state_var'])
            )
            """
            print('xo shape: ', x0.shape, 'X_full_size: ', X_full.shape[0])
            print('x0mu: {}, x0sigma: {}'.format(x0mu, cur[m].traj_info.x0sigma))
            now solve for gmm dynamics by computing the normal inverse wishart prior
            """
            prior = cur[m].traj_info.dynamics.get_prior() # will be DynamicsPriorGMM
            if prior:
                prior.X, prior.U  = X, U
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = sample_size #len(cur_data)
                cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)
            # Fit nonlinear model to normal inverse wishart prior
            
    except KeyboardInterrupt:
        print("shutting down ros")



if __name__ == "__main__":
    main()
