'''
Script to process the San Franscisco data. Currently, parameters must be changed
directly at the script level!

To run the script, use the following command from the parent directory (ie.
    make sure you're in the crime-predictions directory)
    python -m predictor.sf_gp

Authors:
    Alex Wang (alexwang@college.harvard.edu)
    Luis Perez (luis.perez.live@gmail.com)
Copyright 2015, Harvard University
'''

import numpy as np
import pandas as pd
import re
import warnings
import os
import pickle

from . import GP

''' Global Variables '''

sfdata_file = os.path.abspath(
    '../cs281_data/large_data/sfclean.pk')  # Location of data
buckets = 10  # Number of buckets.

# Square Exponential Kernel Parameters
# These are the optimal parameters for n = 10 (we can try with other
# values too)
# BAYESIAN
# l = [9.164520,  0.296120, 10.153288]
# horz = 33.522111
# GPy
l = [0.82754075018, 0.82754075018, 0.82754075018]
horz = 9620.11949755

# This is a function that takes as input the training data results.
# BAYESIAN
# sig_eps_f = lambda train_t: 105.693084
# GPy
sig_eps_f = lambda train_t: train_t.std()

logTransform = False  # Should we do GP under the logspace?

# Prefix to use for plots created in bos directory.
file_prefix = 'GPSEOptimizedGPyTrain'


def createDataMatrix(data):
    '''
    Transforms a panda dataframe into latitude longitude time period matrix
    record of crimes.
    '''
    X = np.zeros((len(data), 3))
    X[:, 1] = data.Latitude.values.astype(float)
    X[:, 2] = data.Longitude.values.astype(float)
    X[:, 0] = data.TimeFeature.values.astype(int)

    return X


def read_data(sfdata_file):
    ''' Read in data '''
    # Let's make a plot for some values of N to see if the data works out...
    with open(sfdata_file) as fp:
        data = pickle.load(fp)
        # For sfdata, need to remove outliers
        data = data[-120 > data.Longitude][data.Longitude > (-130)]
        data = data[data.Latitude > 37][data.Latitude < 40]

    return (createDataMatrix(data))


print "Finished processing San Franscisco data..."

GP.run_gp(read_data(sfdata_file), buckets, l, horz, sig_eps_f, logTransform,
          file_prefix, 'sf')
